#pragma once

#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::point {

/**
 * @brief Inverse mass matrix accessor for explicit time integration in spectral
 * element simulations.
 *
 * This class provides interface for accessing and manipulating
 * inverse mass matrix data at individual quadrature points within spectral
 * elements.
 *
 * @tparam DimensionTag The spatial dimension determining DOF structure:
 *                      - `specfem::dimension::type::dim2`: 2D problems
 * (per-component masses)
 *                      - `specfem::dimension::type::dim3`: 3D problems
 * (per-component masses)
 *
 * @tparam MediumTag The physical medium type affecting mass computation:
 *                   - `specfem::element::medium_tag::acoustic`: Scalar mass for
 * pressure DOF
 *                   - `specfem::element::medium_tag::elastic`: Vector masses
 * for displacement DOFs
 *                   - `specfem::element::medium_tag::poroelastic`: Separate
 * solid and fluid masses
 *
 * @tparam UseSIMD Boolean flag controlling SIMD vectorization for processing
 * multiple quadrature points simultaneously. Critical for performance since
 *                 inverse mass operations dominate explicit time integration
 * costs.
 *
 * @note The diagonal mass matrix property is crucial for the efficiency of
 * spectral element methods and distinguishes them from standard finite element
 * methods which typically have full mass matrices.
 *
 * @see specfem::time_integration::explicit_newmark
 * @see specfem::assembly::mass_matrix
 * @see specfem::quadrature::gll
 * @see specfem::point::acceleration
 *
 * @code
 * // Example: Creating 2D elastic inverse mass matrix for granite
 * using MassInvField = specfem::point::mass_inverse<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     false>;  // Scalar operations for debugging
 *
 * const auto density = 2700.0;        // kg/m³ (granite)
 * const auto jacobian_det = 0.125;    // Element Jacobian determinant
 * const auto quad_weight = 1.0;       // GLL quadrature weight
 *
 * auto mass_value = density * jacobian_det * quad_weight;
 * MassInvField mass_inv(1.0 / mass_value);  // Store inverse
 *
 * // Component-specific masses (if different for x and z)
 * mass_inv(0) = 1.0 / mass_x;  // x-component inverse mass
 * mass_inv(1) = 1.0 / mass_z;  // z-component inverse mass
 * @endcode
 *
 * @code
 * // Example: Explicit time integration using inverse mass matrix
 * MassInvField mass_inverse;
 * AccelField acceleration;
 * ForceField internal_force, external_force;
 *
 * // Compute net force
 * ForceField net_force;
 * for (int comp = 0; comp < net_force.components; ++comp) {
 *   net_force(comp) = external_force(comp) - internal_force(comp);
 * }
 *
 * // Direct acceleration computation (no matrix inversion needed)
 * for (int comp = 0; comp < acceleration.components; ++comp) {
 *   acceleration(comp) = mass_inverse(comp) * net_force(comp);
 * }
 * @endcode
 *
 * @code
 * // Example: SIMD-optimized mass operations for high performance
 * using MassInvFieldSIMD = specfem::point::mass_inverse<
 *     specfem::dimension::type::dim3,
 *     specfem::element::medium_tag::elastic,
 *     true>;   // Enable SIMD for vectorization
 *
 * MassInvFieldSIMD mass_inv_simd;
 *
 * // Process multiple quadrature points simultaneously
 * auto simd_mass_inv = mass_inv_simd.get_simd_data();
 * auto simd_force = force_field.get_simd_data();
 *
 * // Vectorized acceleration computation across SIMD lanes
 * auto simd_acceleration = simd_mass_inv * simd_force;
 * @endcode
 *
 * @code
 * // Example: Mass matrix assembly and inversion
 * const auto properties = element.get_material_properties(quad_point);
 * const auto jacobian = element.get_jacobian_matrix(quad_point);
 * const auto weight = element.get_quadrature_weight(quad_point);
 *
 * // Diagonal mass matrix entry
 * auto mass_entry = properties.density() * jacobian.determinant() * weight;
 *
 * // Store inverse for efficient time integration
 * MassInvField mass_inv;
 * for (int comp = 0; comp < mass_inv.components; ++comp) {
 *   mass_inv(comp) = 1.0 / mass_entry;  // Same for all components in isotropic
 * case
 * }
 * @endcode
 *
 * @code
 * // Example: Critical time step computation
 * MassInvField mass_inverse;
 * StiffnessField diagonal_stiffness;  // Hypothetical diagonal stiffness
 *
 * auto min_dt_squared = std::numeric_limits<type_real>::max();
 * for (int comp = 0; comp < mass_inverse.components; ++comp) {
 *   // dt² ∝ mass/stiffness
 *   auto dt_squared = mass_inverse(comp) / diagonal_stiffness(comp);
 *   min_dt_squared = std::min(min_dt_squared, dt_squared);
 * }
 * auto critical_time_step = std::sqrt(min_dt_squared);
 * @endcode
 *
 * ForceField internal_forces, external_forces;
 *
 * // Load mass matrix and forces
 * specfem::assembly::load_on_device(index, fields, mass_inverse);
 * specfem::assembly::load_on_device(index, forces, internal_forces,
 * external_forces);
 *
 * // Compute acceleration: a = M^(-1) * (F_ext - F_int)
 * for (int icomp = 0; icomp < MassInvField::components; ++icomp) {
 *   acceleration(icomp) = mass_inverse(icomp) *
 *                        (external_forces(icomp) - internal_forces(icomp));
 * }
 *
 * // Store computed acceleration
 * specfem::assembly::store_on_device(index, fields, acceleration);
 * @endcode
 *
 * @see specfem::point::acceleration for acceleration field accessor
 * @see specfem::point::velocity for velocity field accessor
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
class mass_inverse
    : public impl::field<DimensionTag, MediumTag,
                         specfem::data_access::DataClassType::mass_matrix,
                         UseSIMD> {
private:
  /// @brief Type alias for the base field implementation
  using base_type =
      impl::field<DimensionTag, MediumTag,
                  specfem::data_access::DataClassType::mass_matrix, UseSIMD>;

public:
  /// @brief SIMD type for vectorized inverse mass matrix operations
  using simd = typename base_type::simd;

  /// @brief Vector type for storing inverse mass matrix component values
  using value_type = typename base_type::value_type;

  /// @brief Inherit all constructors from the base field implementation
  using base_type::base_type;
};

} // namespace specfem::point
