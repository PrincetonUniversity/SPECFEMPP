#pragma once

#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::point {

/**
 * @brief Point displacement field accessor for spectral element computations.
 *
 * This class provides a specialized interface for accessing and manipulating
 * displacement field data at individual points within spectral elements. It
 * inherits all functionality from the base field implementation while being
 * specifically typed for displacement data.
 *
 * The displacement class is fundamental in wave propagation simulations,
 * representing the spatial displacement of material points from their reference
 * positions. It is commonly used in both elastic and poroelastic media
 * simulations.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3) of the displacement
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
 * // Example: Creating 3D elastic displacement field accessor
 * using DispField = specfem::point::displacement<
 *     specfem::dimension::type::dim3,
 *     specfem::element::medium_tag::elastic,
 *     true>;   // Enable SIMD
 *
 * // Initialize with zero displacement
 * DispField disp(0.0);
 *
 * // Set displacement components
 * disp(0) = 0.001;  // x-component displacement (meters)
 * disp(1) = 0.002;  // y-component displacement
 * disp(2) = 0.003;  // z-component displacement
 *
 * // Use in assembly operations
 * specfem::assembly::store_on_device(point_index, field_container, disp);
 * @endcode
 *
 * @code{.cpp}
 * // Example: Computing strain from displacement gradients
 * DispField displacement;
 * specfem::assembly::load_on_device(index, fields, displacement);
 *
 * // Access displacement components for strain computation
 * auto ux = displacement(0);
 * auto uy = displacement(1);
 * auto uz = displacement(2);
 *
 * // Strain components would be computed from spatial derivatives
 * // (implementation depends on quadrature and differentiation operators)
 * @endcode
 *
 * @see impl::field for inherited functionality
 * @see specfem::point::velocity for velocity field accessor
 * @see specfem::point::acceleration for acceleration field accessor
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
class displacement
    : public impl::field<DimensionTag, MediumTag,
                         specfem::data_access::DataClassType::displacement,
                         UseSIMD> {
private:
  /// @brief Type alias for the base field implementation
  using base_type =
      impl::field<DimensionTag, MediumTag,
                  specfem::data_access::DataClassType::displacement, UseSIMD>;

public:
  /// @brief SIMD type for vectorized displacement operations
  using simd = typename base_type::simd;

  /// @brief Vector type for storing displacement component values
  using value_type = typename base_type::value_type;

  /// @brief Inherit all constructors from the base field implementation
  using base_type::base_type;
};

} // namespace specfem::point
