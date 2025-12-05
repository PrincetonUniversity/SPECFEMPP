#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Stress integrand computation for spectral element weak-form assembly.
 *
 * The stress_integrand class computes the stress-based contributions to the
 * weak-form integral expressions used in spectral element assembly. These
 * integrands represent the stress tensor components transformed to the
 * reference element coordinate system and are essential for computing element
 * matrices and right-hand side vectors.
 *
 * The mathematical formulation varies by medium type:
 *
 * **For elastic domains**, the stress integrand is computed as:
 * \f$
 *   F_{ik} = \sum_{j=1}^{d} T_{ij} \frac{\partial \xi_k}{\partial x_j}
 * \f$
 * where \f$T_{ij}\f$ is the stress tensor, \f$\xi_k\f$ are reference
 * coordinates, and \f$x_j\f$ are physical coordinates. This follows equations
 * (35) & (36) from Komatitsch and Tromp (2002) Part I.
 *
 * **For acoustic domains**, the stress integrand is given by:
 * \f$
 *   F_{ik} = \rho^{-1} \frac{\partial \xi_k}{\partial x_i} \frac{\partial
 * \chi}{\partial \xi_k}
 * \f$
 * where \f$\rho\f$ is density, \f$\chi\f$ is the acoustic potential, following
 * equations (44) & (45) from Komatitsch and Tromp (2002) Part I.
 *
 * @tparam DimensionTag Spatial dimension of the problem (dim2 or dim3).
 * @tparam MediumTag Physical medium type determining stress tensor structure:
 *                  - `acoustic`: Scalar pressure field
 *                  - `elastic`: Full stress tensor components
 *                  - `poroelastic`: Coupled solid-fluid stress components
 * @tparam UseSIMD Boolean flag enabling SIMD vectorization for performance
 *                 optimization across multiple quadrature points.
 *
 * @note The stress integrand computation requires coordinate transformation
 *       derivatives (∂ξ/∂x) typically provided by the Jacobian matrix.
 *
 * @see Komatitsch, D., & Tromp, J. (2002). Spectral-element simulations of
 * global seismic wave propagation—I. Validation. Geophysical Journal
 * International.
 * @see specfem::point::stress for stress tensor representation
 * @see specfem::point::jacobian_matrix for coordinate transformations
 *
 * @code
 * // Example: Computing stress integrand for 2D elastic medium
 * using StressIntegrand = specfem::point::stress_integrand<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     false>;
 *
 * StressIntegrand stress_int;
 *
 * // Load stress tensor and coordinate derivatives
 * specfem::assembly::load_on_device(index, fields, stress_tensor, jacobian);
 *
 * // Compute integrand: F_ik = T_ij * (∂ξ_k/∂x_j)
 * stress_int.compute(stress_tensor, jacobian_inverse);
 *
 * // Use in weak-form assembly
 * element_force += basis_derivative * stress_int * quadrature_weight;
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress_integrand
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::stress_integrand, DimensionTag,
          UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
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
};

} // namespace point
} // namespace specfem
