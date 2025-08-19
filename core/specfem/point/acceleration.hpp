#pragma once

#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::point {

/**
 * @brief Point acceleration field accessor for spectral element computations.
 *
 * This class provides a specialized interface for accessing and manipulating
 * acceleration field data at individual points within spectral elements. It
 * inherits all functionality from the base field implementation while being
 * specifically typed for acceleration data.
 *
 * The acceleration class is commonly used in time-domain wave propagation
 * simulations where acceleration values need to be computed, stored, and
 * accessed at quadrature points during the assembly process.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3) of the acceleration
 * field
 * @tparam MediumTag The medium type (acoustic, elastic, poroelastic, etc.)
 * @tparam UseSIMD Whether to enable SIMD vectorization for performance
 * optimization
 *
 * @note This class inherits all constructors and methods from impl::field,
 * providing component access through operator() and various initialization
 * options.
 *
 * @code{.cpp}
 * // Example: Creating 2D elastic acceleration field accessor
 * using AccelField = specfem::point::acceleration<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     false>;  // No SIMD
 *
 * // Initialize with zero acceleration
 * AccelField accel(0.0);
 *
 * // Set acceleration components
 * accel(0) = 9.81;  // x-component acceleration
 * accel(1) = 0.0;   // z-component acceleration
 *
 * // Use in assembly operations
 * specfem::assembly::load_on_device(point_index, field_container, accel);
 * @endcode
 *
 * @code{.cpp}
 * // Example: Using acceleration in time integration scheme
 * AccelField acceleration;
 * VelocityField velocity;
 * DisplacementField displacement;
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
 * @see impl::field for inherited functionality
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
