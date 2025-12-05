#pragma once

#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::point {

/**
 * @brief Acceleration field accessor
 *
 * This class provides a specialized, type-safe interface for accessing and
 * manipulating acceleration field data at individual quadrature points within
 * spectral elements. Acceleration fields represent the second time derivative
 * of displacement:
 * \f$\mathbf{a} = \ddot{\mathbf{u}} = \frac{\partial^2 \mathbf{u}}{\partial
 * t^2}\f$
 *
 * @tparam DimensionTag The spatial dimension defining acceleration vector size:
 *                      - `specfem::dimension::type::dim2`: 2D problems (ax, az)
 *                      - `specfem::dimension::type::dim3`: 3D problems (ax, ay,
 * az)
 *
 * @tparam MediumTag The physical medium type governing the acceleration
 * computation:
 *                   - `specfem::element::medium_tag::acoustic`: Scalar pressure
 * acceleration
 *                   - `specfem::element::medium_tag::elastic`: Vector
 * displacement acceleration
 *                   - `specfem::element::medium_tag::poroelastic`: Coupled
 * solid-fluid acceleration
 *
 * @tparam UseSIMD Boolean flag controlling SIMD vectorization for processing
 * multiple quadrature points simultaneously. Critical for performance in
 *                 element-wise operations during assembly.
 *
 * @note This class inherits all functionality from the base field
 * implementation while providing specific typing for acceleration data and
 * physics-aware operations.
 *
 * @see specfem::time_integration
 * @see specfem::assembly
 * @see specfem::point::velocity
 * @see specfem::point::displacement
 *
 * @code
 * // Example: Creating 2D elastic acceleration field for earthquake simulation
 * using AccelField = specfem::point::acceleration<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     false>;  // Scalar operations for debugging
 *
 * AccelField accel(0.0);  // Initialize with zero acceleration
 *
 * // Set seismic acceleration components (e.g., from strong ground motion)
 * accel(0) = 2.5;   // Horizontal acceleration (m/s²)
 * accel(1) = -9.81; // Vertical acceleration including gravity (m/s²)
 *
 * // Use in time integration (Newmark scheme example)
 * velocity(0) += accel(0) * dt;              // Update horizontal velocity
 * displacement(0) += velocity(0) * dt + 0.5 * accel(0) * dt * dt;
 * @endcode
 *
 * @code
 * // Example: SIMD-optimized acceleration for high-performance computing
 * using AccelFieldSIMD = specfem::point::acceleration<
 *     specfem::dimension::type::dim3,
 *     specfem::element::medium_tag::elastic,
 *     true>;   // Enable SIMD for multiple points
 *
 * AccelFieldSIMD accel_simd;
 *
 * // Process multiple quadrature points simultaneously
 * auto simd_data = accel_simd.get_simd_data();
 * // Vectorized operations across multiple points
 * simd_data = mass_matrix_inverse * force_vector; // SIMD acceleration
 * computation
 * @endcode
 *
 * @code
 * // Example: Acoustic acceleration for fluid domains
 * using AcousticAccel = specfem::point::acceleration<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::acoustic,
 *     false>;
 *
 * AcousticAccel pressure_accel;
 * // In acoustic media, "acceleration" represents pressure rate of change
 * pressure_accel(0) = bulk_modulus_inv * divergence_velocity;
 * @endcode
 *
 *
 * // Load current values
 * specfem::assembly::load_on_device(index, fields, acceleration, velocity,
 * displacement);
 *
 * // Time integration (Newmark scheme)
 * for (int icomp = 0; icomp < AccelField::components; ++icomp) {
 *   velocity(icomp) += dt * acceleration(icomp);
 *   displacement(icomp) += dt * velocity(icomp) + 0.5 * dt * dt *
 * acceleration(icomp);
 * }
 *
 * // Store updated values
 * specfem::assembly::store_on_device(index, fields, velocity, displacement);
 * @endcode
 *
 * @see specfem::point::velocity for velocity field accessor
 * @see specfem::point::displacement for displacement field accessor
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
class acceleration
    : public impl::field<DimensionTag, MediumTag,
                         specfem::data_access::DataClassType::acceleration,
                         UseSIMD> {
private:
  /// @brief Type alias for the base field implementation
  using base_type =
      impl::field<DimensionTag, MediumTag,
                  specfem::data_access::DataClassType::acceleration, UseSIMD>;

public:
  /// @brief SIMD type for vectorized acceleration operations
  using simd = typename base_type::simd;

  /// @brief Vector type for storing acceleration component values
  using value_type = typename base_type::value_type;

  /// @brief Inherit all constructors from the base field implementation
  using base_type::base_type;
};

} // namespace specfem::point
