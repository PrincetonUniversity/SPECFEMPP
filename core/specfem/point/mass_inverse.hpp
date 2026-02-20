#pragma once

#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::point {

/**
 * @brief Point inverse mass matrix accessor for spectral element computations.
 *
 * This class provides a specialized interface for accessing and manipulating
 * inverse mass matrix data at individual points within spectral elements. It
 * inherits all functionality from the base field implementation while being
 * specifically typed for inverse mass matrix data.
 *
 * The inverse mass matrix is crucial in explicit time integration schemes for
 * wave propagation simulations. It allows for efficient computation of
 * accelerations from forces without requiring matrix inversions during the time
 * stepping process. The diagonal nature of the mass matrix in spectral element
 * methods makes this particularly efficient.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3) of the mass matrix
 * field
 * @tparam MediumTag The medium type (acoustic, elastic, poroelastic, etc.)
 * @tparam UseSIMD Whether to enable SIMD vectorization for performance
 * optimization
 *
 *
 * @code{.cpp}
 * // Example: Creating 2D elastic inverse mass matrix accessor
 * using MassInvField = specfem::point::mass_inverse<
 *     specfem::element::dimension_tag::dim2,
 *     specfem::element::medium_tag::elastic,
 *     false>;  // No SIMD
 *
 * // Initialize with uniform inverse mass
 * MassInvField mass_inv(1.0 / 2700.0);  // 1/density for elastic material
 *
 * // Set component-specific inverse mass values
 * mass_inv(0) = 1.0 / mass_x;  // x-component inverse mass
 * mass_inv(1) = 1.0 / mass_z;  // z-component inverse mass
 *
 * // Use in assembly operations
 * specfem::assembly::load_on_device(point_index, mass_container, mass_inv);
 * @endcode
 *
 * @code{.cpp}
 * // Example: Using inverse mass matrix in explicit time integration
 * MassInvField mass_inverse;
 * AccelField acceleration;
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
 * @see specfem::point::displacement for displacement field accessor
 * @see specfem::point::velocity for velocity field accessor
 * @see specfem::point::acceleration for acceleration field accessor
 */
namespace impl {
template <typename Tags>
class mass_inverse
    : public impl::field<Tags,
                         specfem::data_access::DataClassType::mass_matrix> {
private:
  /**
   * @brief Type alias for the base field implementation.
   */
  using base_type =
      impl::field<Tags, specfem::data_access::DataClassType::mass_matrix>;

public:
  /**
   * @brief SIMD type for vectorized inverse mass matrix operations.
   */
  using simd = typename base_type::simd;

  /**
   * @brief Vector type for storing inverse mass matrix component values.
   */
  using value_type = typename base_type::value_type;

  /**
   * @brief Inherit all constructors from the base field implementation.
   */
  using base_type::base_type;
};
} // namespace impl

template <typename Tags>
using mass_inverse = impl::mass_inverse<specfem::tags::Tags<
    Tags::dimension_tag, Tags::medium_tag, Tags::using_simd> >;

} // namespace specfem::point
