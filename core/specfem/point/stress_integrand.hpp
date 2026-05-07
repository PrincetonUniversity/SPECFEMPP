#pragma once

#include "specfem/data_access.hpp"
#include "specfem/datatype.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Store stress integrands for a quadrature point
 *
 * For elastic domains the stress integrand is given by:
 * \f$ F_{ik} = \sum_{j=1}^{n} T_{ij} \partial_j \xi_{k} \f$ where \f$ T \f$ is
 * the stress tensor. Equation (35) & (36) from Komatitsch and Tromp 2002 I. -
 * Validation
 *
 * For acoustic domains the stress integrand is given by:
 * \f$ F_{ik} = \rho^{-1} \partial_i \xi_{k} \partial_k \chi_{k} \f$. Equation
 * (44) & (45) from Komatitsch and Tromp 2002 I. - Validation
 *
 * @tparam DimensionTag The dimension of the element where the quadrature point
 * is located
 * @tparam MediumTag The medium of the element where the quadrature point is
 * located
 * @tparam UseSIMD Use SIMD instructions
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress_integrand
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::stress_integrand, DimensionTag,
          UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::stress_integrand, DimensionTag,
      UseSIMD>; ///< Base accessor
                ///< type
public:
  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int dimension =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD type
  using value_type =
      typename base_type::template tensor_type<type_real, components,
                                               dimension>; ///< Underlying view
                                                           ///< type to store
                                                           ///< the stress
                                                           ///< integrand
  ///@}

  value_type F; ///< View to store the stress integrand

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION stress_integrand() = default;

  /**
   * @brief Constructor
   *
   * @param F Stress integrands
   */
  KOKKOS_FUNCTION stress_integrand(const value_type &F) : F(F) {}
  ///@}

  /**
   * @name Operators
   */
  ///@{
  /**
   * @brief Transform stress tensor using jacobian matrix.
   *
   * Applies the coordinate transformation from reference element to physical
   * element using the jacobian matrix. This operation transforms stress
   * components from the reference (ξ, ζ) coordinate system to the physical (x,
   * z) coordinate system.
   *
   * The transformation formula for 2D is:
   * \f$ F(i,0) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial x} + T(i,1)
   * \cdot \frac{\partial\zeta}{\partial x}) \f$
   * \f$ F(i,1) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial z} + T(i,1)
   * \cdot \frac{\partial\zeta}{\partial z}) \f$
   *
   * where \f$ J \f$ is the jacobian determinant and the partial derivatives are
   * the inverse jacobian matrix elements.
   *
   * @param jacobian_matrix Jacobian matrix containing transformation
   * derivatives
   * @return Transformed stress tensor in physical coordinates
   *
   * @code
   * stress_integrand_type stress_integrand(transformed_stress_tensor);
   * auto jacobian = compute_jacobian_matrix(quadrature_point);
   * auto stress_tensor = stress_integrand * jacobian;
   * @endcode
   */
  KOKKOS_INLINE_FUNCTION
  value_type operator*(const specfem::point::jacobian_matrix<
                       specfem::element::dimension_tag::dim2, true, UseSIMD>
                           &jacobian_matrix) const {
    value_type T;
    // The Jacobian entries are named xix for \partial\xi / \partial x,
    // but in practice we always call jacobian.inverse(), so xix is actually
    // xxi, xiz is zxi, etc.
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      T(icomponent, 0) =
          jacobian_matrix.jacobian * (F(icomponent, 0) * jacobian_matrix.xix +
                                      F(icomponent, 1) * jacobian_matrix.xiz);
      T(icomponent, 1) = jacobian_matrix.jacobian *
                         (F(icomponent, 0) * jacobian_matrix.gammax +
                          F(icomponent, 1) * jacobian_matrix.gammaz);
    }

    return T;
  }

  /**
   * @brief Transform stress integrand using 3D jacobian matrix.
   *
   * Applies the coordinate transformation from reference element to physical
   * element using the jacobian matrix. This operation transforms stress
   * components from the physical (x, y, z) coordinate system to the reference
   * (ξ, η, γ) coordinate system.
   *
   * The transformation formula for 3D is:
   * \f$ T(i,0) = J \cdot (F(i,0) \cdot \frac{\partial x}{\partial\xi} +
   *                       F(i,1) \cdot \frac{\partial x}{\partial\eta} +
   *                       F(i,2) \cdot \frac{\partial x}{\partial\gamma}) \f$
   * \f$ T(i,1) = J \cdot (F(i,0) \cdot \frac{\partial y}{\partial\xi} +
   *                       F(i,1) \cdot \frac{\partial y}{\partial\eta} +
   *                       F(i,2) \cdot \frac{\partial y}{\partial\gamma}) \f$
   * \f$ T(i,2) = J \cdot (F(i,0) \cdot \frac{\partial z}{\partial\xi} +
   *                       F(i,1) \cdot \frac{\partial z}{\partial\eta} +
   *                       F(i,2) \cdot \frac{\partial z}{\partial\gamma}) \f$
   *
   * where \f$ J \f$ is the inverse jacobian determinant and the partial
   * derivatives are the jacobian matrix elements.
   *
   * @param jacobian_matrix 3D Jacobian matrix containing transformation
   * derivatives
   * @return Untransformed stress tensor
   *
   * @code
   * stress_integrand_type stress_integrand(transformed_stress_tensor);
   * auto jacobian = compute_jacobian_matrix_3d(quadrature_point);
   * auto stress = stress_integrand * jacobian.inverse();
   * @endcode
   */
  KOKKOS_INLINE_FUNCTION
  value_type operator*(const specfem::point::jacobian_matrix<
                       specfem::element::dimension_tag::dim3, true, UseSIMD>
                           &jacobian_matrix) const {
    value_type T;

    // The Jacobian entries are named xix for \partial\xi / \partial x,
    // but in practice we always call jacobian.inverse(), so xix is actually
    // xxi, xiy is yxi, etc.
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      T(icomponent, 0) =
          jacobian_matrix.jacobian * (F(icomponent, 0) * jacobian_matrix.xix +
                                      F(icomponent, 1) * jacobian_matrix.xiy +
                                      F(icomponent, 2) * jacobian_matrix.xiz);
      T(icomponent, 1) =
          jacobian_matrix.jacobian * (F(icomponent, 0) * jacobian_matrix.etax +
                                      F(icomponent, 1) * jacobian_matrix.etay +
                                      F(icomponent, 2) * jacobian_matrix.etaz);
      T(icomponent, 2) = jacobian_matrix.jacobian *
                         (F(icomponent, 0) * jacobian_matrix.gammax +
                          F(icomponent, 1) * jacobian_matrix.gammay +
                          F(icomponent, 2) * jacobian_matrix.gammaz);
    }

    return T;
  }
  ///@}
};

} // namespace point
} // namespace specfem
