#pragma once

#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::point {

/**
 * @brief Point velocity field accessor for spectral element computations.
 *
 * This class provides a specialized interface for accessing and manipulating
 * velocity field data at individual points within spectral elements. It
 * inherits all functionality from the base field implementation while being
 * specifically typed for velocity data.
 *
 * The velocity class represents the time derivative of displacement and is
 * essential in time-domain wave propagation simulations. It is used in time
 * integration schemes and for computing kinetic energy in the system.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3) of the velocity
 * field
 * @tparam MediumTag The medium type (acoustic, elastic, poroelastic, etc.)
 * @tparam UseSIMD Whether to enable SIMD vectorization for performance
 * optimization
 *
 *
 * @code{.cpp}
 * // Example: Creating 2D acoustic velocity field accessor
 * using VelField = specfem::point::velocity<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::acoustic,
 *     false>;  // No SIMD
 *
 * // Initialize with zero velocity
 * VelField vel(0.0);
 *
 * // Set velocity components
 * vel(0) = 0.1;   // x-component velocity (m/s)
 * vel(1) = -0.05; // z-component velocity
 *
 * // Use in assembly operations
 * specfem::assembly::load_on_device(point_index, field_container, vel);
 * @endcode
 *
 * @code{.cpp}
 * // Example: Velocity in time integration (Newmark scheme)
 * VelField velocity_old, velocity_new;
 * AccelField acceleration;
 * const double dt = 0.001; // time step
 * const double gamma = 0.5; // Newmark parameter
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
