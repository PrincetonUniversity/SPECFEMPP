#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Field gradient computation for spectral element variational
 * formulations.
 *
 * The field_derivatives class computes and stores spatial derivatives of field
 * variables at quadrature points within spectral elements. These derivatives
 * are fundamental to implementing weak-form partial differential equations in
 * spectral element methods, enabling the computation of strain tensors,
 * divergence operators, and other differential quantities essential for wave
 * propagation simulations.
 *
 * **Mathematical Foundation:**
 * The field derivatives represent the spatial gradient tensor:
 * \f$
 *   \nabla \mathbf{u} = \left[ \frac{\partial u_i}{\partial x_j} \right]_{i,j}
 * \f$
 * where \f$u_i\f$ are the field components and \f$x_j\f$ are spatial
 * coordinates.
 *
 * For specific media, these derivatives enable computation of:
 * - **Elastic media**: Strain tensor \f$\epsilon_{ij} = \frac{1}{2}(\partial_i
 * u_j + \partial_j u_i)\f$
 * - **Acoustic media**: Velocity divergence \f$\nabla \cdot \mathbf{v}\f$ for
 * pressure computation
 * - **Poroelastic media**: Solid and fluid phase strain components
 *
 * @tparam DimensionTag Spatial dimension determining gradient tensor size:
 *                      - `dim2`: 2×2 gradient matrix for 2D problems
 *                      - `dim3`: 3×3 gradient matrix for 3D problems
 * @tparam MediumTag Physical medium type determining field interpretation:
 *                  - `acoustic`: Velocity potential derivatives
 *                  - `elastic`: Displacement gradient tensor
 *                  - `poroelastic`: Solid and fluid displacement gradients
 * @tparam UseSIMD Boolean flag enabling SIMD vectorization for processing
 *                 multiple field derivatives simultaneously.
 *
 * @note Field derivatives are typically computed by transforming
 * reference-space derivatives using the Jacobian matrix of coordinate
 * transformation.
 *
 * @see specfem::point::stress for stress tensor computation
 * @see specfem::point::jacobian_matrix for coordinate transformations
 * @see specfem::assembly for weak-form integration procedures
 *
 * @code
 * // Example: Computing strain tensor from displacement derivatives
 * using FieldDerivatives = specfem::point::field_derivatives<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     false>;
 *
 * FieldDerivatives grad_u;
 * specfem::assembly::load_on_device(index, fields, grad_u);
 *
 * // Compute strain tensor: ε = (∇u + (∇u)^T) / 2
 * auto epsilon_xx = grad_u(0, 0);  // ∂u_x/∂x
 * auto epsilon_zz = grad_u(1, 1);  // ∂u_z/∂z
 * auto epsilon_xz = 0.5 * (grad_u(0, 1) + grad_u(1, 0));  // Shear strain
 *
 * // Use in stress computation
 * auto stress = constitutive_relation(epsilon_xx, epsilon_zz, epsilon_xz);
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct field_derivatives
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::field_derivatives, DimensionTag,
          UseSIMD> {

private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::field_derivatives, DimensionTag,
      UseSIMD>; ///< Base type of the
                ///< point field
                ///< derivatives
public:
  /**
   * @name Compile time constants
   *
   */
  ///@{
  static constexpr int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  constexpr static auto medium_tag = MediumTag; ///< Medium tag for the element
  constexpr static int num_dimensions =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD data type
  using value_type =
      typename base_type::template tensor_type<type_real, components,
                                               num_dimensions>;
  ///@}

  value_type du; ///< View to store the field derivatives.

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION field_derivatives() = default;

  /**
   * @brief Constructor
   *
   * @param du Field derivatives
   */
  KOKKOS_FUNCTION field_derivatives(const value_type &du) : du(du) {}
  ///@}
};

} // namespace point
} // namespace specfem
