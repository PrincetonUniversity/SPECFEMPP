#pragma once

#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::point {

/**
 * @brief Quadrature weights container for spectral element numerical
 * integration.
 *
 * The weights struct stores the quadrature weights associated with
 * Gauss-Lobatto-Legendre (GLL) or Gauss-Legendre-Chebyshev quadrature points
 * within spectral elements. These weights are fundamental to the numerical
 * integration process in spectral element methods, representing the
 * differential area/volume elements for accurate integration of weak-form
 * equations.
 *
 * In spectral element methods, integrals over elements are computed using
 * quadrature rules:
 * \f$
 *   \int_{\Omega_e} f(\mathbf{x}) \, d\mathbf{x} \approx \sum_{i,j}
 * f(\mathbf{x}_{ij}) \cdot w_i \cdot w_j \cdot J_{ij}
 * \f$
 * where \f$w_i, w_j\f$ are the quadrature weights, \f$\mathbf{x}_{ij}\f$ are
 * quadrature points, and \f$J_{ij}\f$ is the Jacobian of the transformation.
 *
 * @tparam DimensionTag Spatial dimension of the element (dim2 for 2D, dim3 for
 * 3D). Determines the number of weight components stored.
 *
 * @note Quadrature weights are typically precomputed during mesh generation and
 * remain constant throughout the simulation for elements with fixed geometry.
 *
 * @see specfem::quadrature for quadrature rule computation
 * @see specfem::point::jacobian_matrix for geometric transformation
 *
 * @code
 * // Example: Computing element integral using weights
 * specfem::point::weights<specfem::dimension::type::dim2> w;
 * specfem::assembly::load_on_device(index, quadrature, w);
 *
 * // Numerical integration over 2D element
 * type_real element_integral = function_value * w.wx * w.wz * jacobian_det;
 * @endcode
 */
template <specfem::dimension::type DimensionTag> struct weights;

/**
 * @brief 2D specialization for quadrature weights in spectral elements.
 *
 * This specialization stores the quadrature weights for 2D spectral elements,
 * containing weights in both the x and z directions (wx, wz). These weights
 * correspond to the Gauss-Lobatto-Legendre (GLL) or Gauss-Legendre quadrature
 * rules used for numerical integration within each element.
 *
 * The 2D quadrature weight contribution to an integral is computed as the
 * tensor product:
 * \f$
 *   w_{total} = w_x \cdot w_z
 * \f$
 * where \f$w_x\f$ and \f$w_z\f$ are the 1D quadrature weights in their
 * respective directions.
 *
 * These weights are essential for:
 * - Mass matrix assembly: \f$M_{ij} = \int \phi_i \phi_j \, d\Omega\f$
 * - Stiffness matrix assembly: \f$K_{ij} = \int \nabla\phi_i \cdot \nabla\phi_j
 * \, d\Omega\f$
 * - Source term integration: \f$f_i = \int \phi_i f \, d\Omega\f$
 * - Energy and norm computations
 *
 * @note The weights are typically computed once during preprocessing and stored
 *       for efficient access during assembly operations.
 *
 * @see specfem::quadrature::gll for GLL quadrature rule generation
 * @see specfem::point::jacobian_matrix for coordinate transformation
 *
 * @code
 * // Example: 2D mass matrix element assembly
 * specfem::point::weights<specfem::dimension::type::dim2> weights;
 * specfem::assembly::load_on_device(point_index, quadrature_data, weights);
 *
 * // Compute quadrature contribution
 * type_real quad_weight = weights.wx * weights.wz;
 * type_real mass_contribution = basis_i * basis_j * quad_weight * jacobian;
 *
 * // Add to global mass matrix
 * global_mass(iglob_i, iglob_j) += mass_contribution;
 * @endcode
 */
template <>
struct weights<specfem::dimension::type::dim2>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::weights,
          specfem::dimension::type::dim2, false> {
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Compile-time dimension identifier
                                      ///< for template specialization dispatch

  type_real wz; ///< Quadrature weight in the z-direction (vertical)
                ///< corresponding to the GLL/Gauss point. Used in
                ///< tensor-product integration. Typically ranges from 0 to 2
                ///< depending on quadrature rule.

  type_real wx; ///< Quadrature weight in the x-direction (horizontal)
                ///< corresponding to the GLL/Gauss point. Used in
                ///< tensor-product integration. Combined with wz for 2D
                ///< integration: w_total = wx * wz.

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_INLINE_FUNCTION
  weights() = default;

  /**
   * @brief Constructor with values
   *
   */
  KOKKOS_INLINE_FUNCTION
  weights(const type_real &wz, const type_real &wx) : wz(wz), wx(wx) {}
  ///@}

  /**
   * @brief Get the product of the weights
   *
   * @return type_real Product of the weights
   */
  KOKKOS_INLINE_FUNCTION type_real product() const { return wz * wx; }
};

template <>
struct weights<specfem::dimension::type::dim3>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::weights,
          specfem::dimension::type::dim3, false> {
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag
  type_real wz; ///< Weight of the quadrature point in the z direction within
                ///< the spectral element
  type_real wy; ///< Weight of the quadrature point in the y direction within
                ///< the spectral element
  type_real wx; ///< Weight of the quadrature point in the x direction within
                ///< the spectral element

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_INLINE_FUNCTION
  weights() = default;

  /**
   * @brief Constructor with values
   *
   */
  KOKKOS_INLINE_FUNCTION
  weights(const type_real &wz, const type_real &wy, const type_real &wx)
      : wz(wz), wy(wy), wx(wx) {}
  ///@}

  /**
   * @brief Get the product of the weights
   *
   * @return type_real Product of the weights
   */
  KOKKOS_INLINE_FUNCTION type_real product() const { return wz * wy * wx; }
};

} // namespace specfem::point
