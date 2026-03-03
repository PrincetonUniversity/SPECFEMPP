#pragma once

#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include "specfem/point/field_derivatives.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Symmetric strain tensor at a quadrature point.
 *
 * Stores the independent components of the strain tensor
 * \f$ \varepsilon_{ij} = \tfrac{1}{2}(\partial_i u_j + \partial_j u_i) \f$,
 * computed from a @ref specfem::point::field_derivatives object.
 *
 * This is a purely kinematic quantity associated with the displacement
 * field and is independent of any attenuation model.
 *
 * Helper methods return the volumetric trace
 * \f$\varepsilon_{kk}\f$ and the deviatoric part
 * \f$\varepsilon_{ij}^\text{dev} = \varepsilon_{ij} -
 *    \tfrac{1}{3}\varepsilon_{kk}\delta_{ij}\f$.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam MediumTag    Medium type (elastic_psv, elastic, …)
 * @tparam UseSIMD      Enable SIMD vectorization
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct strain;

// ---------------------------------------------------------------------------
// dim2 / elastic_psv  specialization
// ---------------------------------------------------------------------------

/**
 * @brief Strain tensor for 2D PSV elastic elements.
 *
 * Stores the three independent in-plane strain components:
 * - \f$\varepsilon_{xx} = \partial u_x / \partial x\f$
 * - \f$\varepsilon_{zz} = \partial u_z / \partial z\f$
 * - \f$\varepsilon_{xz} = \tfrac{1}{2}(\partial u_x/\partial z +
 *                                       \partial u_z/\partial x)\f$
 *
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <bool UseSIMD>
struct strain<specfem::element::dimension_tag::dim2,
              specfem::element::medium_tag::elastic_psv, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::strain,
          specfem::element::dimension_tag::dim2, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::strain,
      specfem::element::dimension_tag::dim2, UseSIMD>;

public:
  // -----------------------------------------------------------------------
  // Static properties
  // -----------------------------------------------------------------------
  constexpr static specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim2;
  constexpr static specfem::element::medium_tag medium_tag =
      specfem::element::medium_tag::elastic_psv;
  constexpr static bool using_simd = UseSIMD;

  // -----------------------------------------------------------------------
  // Type aliases
  // -----------------------------------------------------------------------
  using simd = typename base_type::template simd<type_real>;
  using scalar_type = typename base_type::template scalar_type<type_real>;

  // -----------------------------------------------------------------------
  // Data members
  // -----------------------------------------------------------------------
  scalar_type epsilon_xx; ///< Normal strain \f$\varepsilon_{xx}\f$
  scalar_type epsilon_zz; ///< Normal strain \f$\varepsilon_{zz}\f$
  scalar_type epsilon_xz; ///< Shear strain  \f$\varepsilon_{xz}\f$ (tensor)

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------

  /** @brief Default constructor — zeroes all fields via init(). */
  KOKKOS_FUNCTION strain() { this->init(); }

  /**
   * @brief Component-wise value constructor.
   *
   * @param epsilon_xx Normal strain \f$\varepsilon_{xx}\f$
   * @param epsilon_zz Normal strain \f$\varepsilon_{zz}\f$
   * @param epsilon_xz Shear strain  \f$\varepsilon_{xz}\f$ (tensor convention)
   */
  KOKKOS_FUNCTION strain(const scalar_type &epsilon_xx,
                         const scalar_type &epsilon_zz,
                         const scalar_type &epsilon_xz)
      : epsilon_xx(epsilon_xx), epsilon_zz(epsilon_zz), epsilon_xz(epsilon_xz) {
  }

  /**
   * @brief Construct strain from displacement field derivatives.
   *
   * Computes:
   * \f{align}{
   *   \varepsilon_{xx} &= \mathtt{du}(0,0) \\
   *   \varepsilon_{zz} &= \mathtt{du}(1,1) \\
   *   \varepsilon_{xz} &= \tfrac{1}{2}(\mathtt{du}(0,1) + \mathtt{du}(1,0))
   * \f}
   * where \f$\mathtt{du}(k,i) = \partial u_k / \partial x_i\f$.
   *
   * @param fd Field derivatives at this quadrature point
   */
  KOKKOS_FUNCTION explicit strain(
      const specfem::point::field_derivatives<
          specfem::element::dimension_tag::dim2,
          specfem::element::medium_tag::elastic_psv, UseSIMD> &fd)
      : epsilon_xx(fd.du(0, 0)), epsilon_zz(fd.du(1, 1)),
        epsilon_xz(type_real(0.5) * (fd.du(0, 1) + fd.du(1, 0))) {}

  // -----------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------

  /** @brief Zero all strain fields. */
  KOKKOS_FUNCTION void init() {
    this->epsilon_xx = scalar_type(0);
    this->epsilon_zz = scalar_type(0);
    this->epsilon_xz = scalar_type(0);
  }

  /**
   * @brief Volumetric strain trace \f$\varepsilon_{kk} =
   *        \varepsilon_{xx} + \varepsilon_{zz}\f$.
   */
  KOKKOS_INLINE_FUNCTION scalar_type trace() const {
    return epsilon_xx + epsilon_zz;
  }

  /**
   * @brief Deviatoric strain \f$\varepsilon_{ij}^\text{dev} =
   *        \varepsilon_{ij} - \tfrac{1}{3}\varepsilon_{kk}\delta_{ij}\f$.
   *
   * Off-diagonal components are unaffected; each normal component has
   * \f$\varepsilon_{kk}/3\f$ subtracted.
   */
  KOKKOS_INLINE_FUNCTION strain deviatoric() const {
    const scalar_type third_trace = type_real(1.0 / 3.0) * trace();
    return strain(epsilon_xx - third_trace, epsilon_zz - third_trace,
                  epsilon_xz);
  }

  /** @brief Equality operator. */
  KOKKOS_INLINE_FUNCTION bool operator==(const strain &other) const {
    return epsilon_xx == other.epsilon_xx && epsilon_zz == other.epsilon_zz &&
           epsilon_xz == other.epsilon_xz;
  }
};

// ---------------------------------------------------------------------------
// dim3 / elastic  specialization
// ---------------------------------------------------------------------------

/**
 * @brief Strain tensor for 3D elastic elements.
 *
 * Stores the six independent strain components:
 * - \f$\varepsilon_{xx} = \partial u_x/\partial x\f$
 * - \f$\varepsilon_{yy} = \partial u_y/\partial y\f$
 * - \f$\varepsilon_{zz} = \partial u_z/\partial z\f$
 * - \f$\varepsilon_{xy} = \tfrac{1}{2}(\partial u_x/\partial y +
 *                                       \partial u_y/\partial x)\f$
 * - \f$\varepsilon_{xz} = \tfrac{1}{2}(\partial u_x/\partial z +
 *                                       \partial u_z/\partial x)\f$
 * - \f$\varepsilon_{yz} = \tfrac{1}{2}(\partial u_y/\partial z +
 *                                       \partial u_z/\partial y)\f$
 *
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <bool UseSIMD>
struct strain<specfem::element::dimension_tag::dim3,
              specfem::element::medium_tag::elastic, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::strain,
          specfem::element::dimension_tag::dim3, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::strain,
      specfem::element::dimension_tag::dim3, UseSIMD>;

public:
  // -----------------------------------------------------------------------
  // Static properties
  // -----------------------------------------------------------------------
  constexpr static specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim3;
  constexpr static specfem::element::medium_tag medium_tag =
      specfem::element::medium_tag::elastic;
  constexpr static bool using_simd = UseSIMD;

  // -----------------------------------------------------------------------
  // Type aliases
  // -----------------------------------------------------------------------
  using simd = typename base_type::template simd<type_real>;
  using scalar_type = typename base_type::template scalar_type<type_real>;

  // -----------------------------------------------------------------------
  // Data members
  // -----------------------------------------------------------------------
  scalar_type epsilon_xx; ///< Normal strain \f$\varepsilon_{xx}\f$
  scalar_type epsilon_yy; ///< Normal strain \f$\varepsilon_{yy}\f$
  scalar_type epsilon_zz; ///< Normal strain \f$\varepsilon_{zz}\f$
  scalar_type epsilon_xy; ///< Shear strain  \f$\varepsilon_{xy}\f$ (tensor)
  scalar_type epsilon_xz; ///< Shear strain  \f$\varepsilon_{xz}\f$ (tensor)
  scalar_type epsilon_yz; ///< Shear strain  \f$\varepsilon_{yz}\f$ (tensor)

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------

  /** @brief Default constructor — zeroes all fields via init(). */
  KOKKOS_FUNCTION strain() { this->init(); }

  /**
   * @brief Component-wise value constructor.
   *
   * @param epsilon_xx Normal strain \f$\varepsilon_{xx}\f$
   * @param epsilon_yy Normal strain \f$\varepsilon_{yy}\f$
   * @param epsilon_zz Normal strain \f$\varepsilon_{zz}\f$
   * @param epsilon_xy Shear strain  \f$\varepsilon_{xy}\f$ (tensor convention)
   * @param epsilon_xz Shear strain  \f$\varepsilon_{xz}\f$ (tensor convention)
   * @param epsilon_yz Shear strain  \f$\varepsilon_{yz}\f$ (tensor convention)
   */
  KOKKOS_FUNCTION
  strain(const scalar_type &epsilon_xx, const scalar_type &epsilon_yy,
         const scalar_type &epsilon_zz, const scalar_type &epsilon_xy,
         const scalar_type &epsilon_xz, const scalar_type &epsilon_yz)
      : epsilon_xx(epsilon_xx), epsilon_yy(epsilon_yy), epsilon_zz(epsilon_zz),
        epsilon_xy(epsilon_xy), epsilon_xz(epsilon_xz), epsilon_yz(epsilon_yz) {
  }

  /**
   * @brief Construct strain from displacement field derivatives.
   *
   * Computes:
   * \f{align}{
   *   \varepsilon_{xx} &= \mathtt{du}(0,0), \quad
   *   \varepsilon_{yy}  = \mathtt{du}(1,1), \quad
   *   \varepsilon_{zz}  = \mathtt{du}(2,2) \\
   *   \varepsilon_{xy} &= \tfrac{1}{2}(\mathtt{du}(0,1) + \mathtt{du}(1,0)) \\
   *   \varepsilon_{xz} &= \tfrac{1}{2}(\mathtt{du}(0,2) + \mathtt{du}(2,0)) \\
   *   \varepsilon_{yz} &= \tfrac{1}{2}(\mathtt{du}(1,2) + \mathtt{du}(2,1))
   * \f}
   * where \f$\mathtt{du}(k,i) = \partial u_k / \partial x_i\f$.
   *
   * @param fd Field derivatives at this quadrature point
   */
  KOKKOS_FUNCTION explicit strain(
      const specfem::point::field_derivatives<
          specfem::element::dimension_tag::dim3,
          specfem::element::medium_tag::elastic, UseSIMD> &fd)
      : epsilon_xx(fd.du(0, 0)), epsilon_yy(fd.du(1, 1)),
        epsilon_zz(fd.du(2, 2)),
        epsilon_xy(type_real(0.5) * (fd.du(0, 1) + fd.du(1, 0))),
        epsilon_xz(type_real(0.5) * (fd.du(0, 2) + fd.du(2, 0))),
        epsilon_yz(type_real(0.5) * (fd.du(1, 2) + fd.du(2, 1))) {}

  // -----------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------

  /** @brief Zero all strain fields. */
  KOKKOS_FUNCTION void init() {
    this->epsilon_xx = scalar_type(0);
    this->epsilon_yy = scalar_type(0);
    this->epsilon_zz = scalar_type(0);
    this->epsilon_xy = scalar_type(0);
    this->epsilon_xz = scalar_type(0);
    this->epsilon_yz = scalar_type(0);
  }

  /**
   * @brief Volumetric strain trace \f$\varepsilon_{kk} =
   *        \varepsilon_{xx} + \varepsilon_{yy} + \varepsilon_{zz}\f$.
   */
  KOKKOS_INLINE_FUNCTION scalar_type trace() const {
    return epsilon_xx + epsilon_yy + epsilon_zz;
  }

  /**
   * @brief Deviatoric strain \f$\varepsilon_{ij}^\text{dev} =
   *        \varepsilon_{ij} - \tfrac{1}{3}\varepsilon_{kk}\delta_{ij}\f$.
   *
   * Off-diagonal (shear) components are unaffected; each normal component
   * has \f$\varepsilon_{kk}/3\f$ subtracted.
   */
  KOKKOS_INLINE_FUNCTION strain deviatoric() const {
    const scalar_type third_trace = type_real(1.0 / 3.0) * trace();
    return strain(epsilon_xx - third_trace, epsilon_yy - third_trace,
                  epsilon_zz - third_trace, epsilon_xy, epsilon_xz, epsilon_yz);
  }

  /** @brief Equality operator. */
  KOKKOS_INLINE_FUNCTION bool operator==(const strain &other) const {
    return epsilon_xx == other.epsilon_xx && epsilon_yy == other.epsilon_yy &&
           epsilon_zz == other.epsilon_zz && epsilon_xy == other.epsilon_xy &&
           epsilon_xz == other.epsilon_xz && epsilon_yz == other.epsilon_yz;
  }
};

} // namespace point
} // namespace specfem
