#pragma once

#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::point {

/**
 * @brief Velocity field accessor for spectral element wave propagation and
 * dynamic analysis.
 *
 * This class provides a type-safe, high-performance interface for accessing and
 * manipulating velocity field data at individual quadrature points within
 * spectral elements. Velocity
 * \f$\mathbf{v} = \dot{\mathbf{u}}\f$ represents the first time derivative of
 * displacement and is fundamental to dynamic analysis and wave propagation
 * simulations.
 *
 * @tparam DimensionTag The spatial dimension defining velocity vector size:
 *                      - `specfem::dimension::type::dim2`: 2D problems (vx, vz)
 *                      - `specfem::dimension::type::dim3`: 3D problems (vx, vy,
 * vz)
 *
 * @tparam MediumTag The physical medium type governing velocity behavior:
 *                   - `specfem::element::medium_tag::acoustic`: Pressure wave
 * velocity (scalar)
 *                   - `specfem::element::medium_tag::elastic`: Particle
 * velocity (vector)
 *                   - `specfem::element::medium_tag::poroelastic`: Separate
 * solid and fluid velocities
 *
 * @tparam UseSIMD Boolean flag controlling SIMD vectorization for processing
 * multiple quadrature points simultaneously. Critical for performance in
 *                 time-marching algorithms where velocity updates dominate
 * computation.
 *
 * @note In explicit time integration, velocity updates are typically the most
 *       computationally intensive operations, making SIMD optimization
 * essential.
 *
 * @see specfem::point::displacement
 * @see specfem::point::acceleration
 * @see specfem::time_integration
 * @see specfem::boundary_conditions::stacey
 *
 * @code
 * // Example: Creating 2D elastic velocity field for seismic simulation
 * using VelField = specfem::point::velocity<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     false>;  // Scalar operations for debugging
 *
 * VelField vel(0.0);  // Initialize with zero velocity
 *
 * // Set velocity components (e.g., from initial conditions)
 * vel(0) = 0.1;   // 10 cm/s horizontal velocity
 * vel(1) = -0.05; // -5 cm/s vertical velocity
 *
 * // Compute kinetic energy density
 * const auto density = 2500.0;  // kg/m³ for typical rock
 * auto ke_density = 0.5 * density * (vel(0)*vel(0) + vel(1)*vel(1));
 * @endcode
 *
 * @code
 * // Example: Newmark time integration with velocity
 * VelField velocity_old, velocity_new;
 * AccelField acceleration;
 * DispField displacement;
 *
 * const auto dt = 0.001;    // 1 ms time step
 * const auto gamma = 0.5;   // Newmark parameter (average acceleration)
 *
 * // Velocity update in Newmark scheme
 * for (int comp = 0; comp < velocity_old.components; ++comp) {
 *   velocity_new(comp) = velocity_old(comp) +
 *     (1.0 - gamma) * acceleration_old(comp) * dt +
 *     gamma * acceleration_new(comp) * dt;
 * }
 * @endcode
 *
 * @code
 * // Example: Acoustic pressure wave velocity (1D)
 * using AcousticVel = specfem::point::velocity<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::acoustic,
 *     true>;   // Enable SIMD for performance
 *
 * AcousticVel pressure_vel;
 *
 * // In acoustic formulation, "velocity" often represents pressure time
 * derivative
 * // Related to actual particle velocity through: v_particle =
 * -grad(pressure)/(rho*omega^2) pressure_vel(0) = bulk_modulus_inv *
 * pressure_rate;
 * @endcode
 *
 * @code
 * // Example: Boundary condition enforcement
 * VelField velocity;
 *
 * // Stacey absorbing boundary condition
 * if (is_stacey_boundary(point_index)) {
 *   auto normal = get_boundary_normal(point_index);
 *   auto impedance = get_acoustic_impedance(point_index);
 *
 *   // Apply absorbing condition: traction = -impedance * velocity
 *   auto normal_velocity = velocity(0) * normal[0] + velocity(1) * normal[1];
 *   auto absorbing_traction = -impedance * normal_velocity;
 *   apply_boundary_traction(absorbing_traction, normal);
 * }
 * @endcode
 *
 * @code
 * // Example: Energy and momentum analysis
 * VelField velocity;
 * const auto density = material_properties.density();
 * const auto volume_element = jacobian.determinant() * quadrature_weight;
 *
 * // Total kinetic energy at this point
 * auto velocity_magnitude_squared = 0.0;
 * for (int comp = 0; comp < velocity.components; ++comp) {
 *   velocity_magnitude_squared += velocity(comp) * velocity(comp);
 * }
 * auto kinetic_energy = 0.5 * density * velocity_magnitude_squared *
 * volume_element;
 *
 * // Momentum components
 * auto momentum_x = density * velocity(0) * volume_element;
 * auto momentum_z = density * velocity(1) * volume_element;
 * @endcode
 *
 *
 * // Load current values
 * specfem::assembly::load_on_device(index, fields, velocity_old, acceleration);
 *
 * // Update velocity using Newmark scheme
 * for (int icomp = 0; icomp < VelField::components; ++icomp) {
 *   velocity_new(icomp) = velocity_old(icomp) + dt * gamma *
 * acceleration(icomp);
 * }
 *
 * // Store updated velocity
 * specfem::assembly::store_on_device(index, fields, velocity_new);
 * @endcode
 *
 * @see specfem::point::displacement for displacement field accessor
 * @see specfem::point::acceleration for acceleration field accessor
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
class velocity
    : public impl::field<DimensionTag, MediumTag,
                         specfem::data_access::DataClassType::velocity,
                         UseSIMD> {
private:
  /// @brief Type alias for the base field implementation
  using base_type =
      impl::field<DimensionTag, MediumTag,
                  specfem::data_access::DataClassType::velocity, UseSIMD>;

public:
  /// @brief SIMD type for vectorized velocity operations
  using simd = typename base_type::simd;

  /// @brief Vector type for storing velocity component values
  using value_type = typename base_type::value_type;

  /// @brief Inherit all constructors from the base field implementation
  using base_type::base_type;
};

} // namespace specfem::point
