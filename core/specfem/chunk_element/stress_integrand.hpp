#pragma once

#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace chunk_element {

/**
 * @brief Struct to hold the stress integrand at every quadrature point within a
 * chunk of elements.
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
 * @tparam NumberElements Number of elements in the chunk.
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points.
 * @tparam DimensionTag Dimension type for elements within the chunk.
 * @tparam MediumTag Medium tag for elements within the chunk.
 * @tparam MemorySpace Memory space for data storage.
 * @tparam MemoryTraits Memory traits for data storage.
 * @tparam UseSIMD Flag to indicate if SIMD should be used.
 */
template <int NumberElements, int NGLL,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool UseSIMD>
struct stress_integrand {

public:
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static int num_elements =
      NumberElements; ///< Number of elements in the chunk.
  constexpr static auto dimension =
      DimensionTag; ///< Dimension type for elements.
  ///@}

private:
  constexpr static int num_dimensions =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::dimension; ///< Number of
  ///< dimensions.
  constexpr static int components =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::components; ///< Number of
                                                           ///< components.

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

  using ViewType = specfem::datatype::TensorChunkElementViewType<
      type_real, DimensionTag, NumberElements, NGLL, components, num_dimensions,
      UseSIMD, MemorySpace,
      MemoryTraits>; ///< Underlying view used to store data.
  ///@}

  ViewType F; ///< Stress integrand

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor.
   *
   */
  KOKKOS_FUNCTION stress_integrand() = default;

  /**
   * @brief Constructor that initializes the stress integrand with a given view.
   *
   * @param F Stress integrand view.
   */
  KOKKOS_FUNCTION stress_integrand(const ViewType &F) : F(F) {}

  /**
   * @brief Constructor that initializes the stress integrand within Scratch
   * Memory.
   *
   * @tparam MemberType Kokos team member type.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_FUNCTION stress_integrand(const MemberType &team)
      : F(team.team_scratch(0)) {}

  ///@}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int  Amount of shared memory required
   */
  constexpr static int shmem_size() { return ViewType::shmem_size(); }

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
   */
  KOKKOS_INLINE_FUNCTION
  ViewType operator*(const specfem::point::jacobian_matrix<
                     specfem::element::dimension_tag::dim2, true, UseSIMD>
                         &jacobian_matrix) const {
    ViewType T;
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
   */
  KOKKOS_INLINE_FUNCTION
  ViewType operator*(const specfem::point::jacobian_matrix<
                     specfem::element::dimension_tag::dim3, true, UseSIMD>
                         &jacobian_matrix) const {
    ViewType T;

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

} // namespace chunk_element
} // namespace specfem
