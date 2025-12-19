#pragma once

#include "impl/atomic_add_access_functions.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/atomic_add_access_functions.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

namespace fields_impl {
template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              ((specfem::data_access::is_index_type<IndexType>::value) &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
atomic_add_on_device(const IndexType &index, const ContainerType &field,
                     AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  using simd_accessor_type =
      std::integral_constant<bool, IndexType::using_simd>;
  simulation_field_impl::atomic_add_after_simd_dispatch<true>(
      simd_accessor_type(), index, field, accessors...);
  return;
}
} // namespace fields_impl

/**
 * @brief Atomic field accumulation for spectral elements on device.
 *
 * Public interface for atomically accumulating values into simulation fields on
 * GPU devices. This function provides a unified interface for adding values to
 * multiple field components simultaneously with guaranteed thread-safety
 * through atomic operations.
 *
 * @ingroup FieldsDataAccess
 *
 * @tparam IndexType Index type (specfem::point::index with SIMD support)
 * @tparam ContainerType Simulation field container (2D/3D specializations)
 * @tparam AccessorTypes Variadic field accessor types
 * (specfem::point::displacement, specfem::point::velocity,
 * specfem::point::acceleration)
 *
 * @param index Spatial index (element + quadrature point information with SIMD
 * support)
 * @param field Simulation field container holding medium-specific field data
 * @param accessors Variable number of field accessors for simultaneous atomic
 * updates
 *
 * @pre All accessors must have the same medium tag (e.g., all elastic or all
 * acoustic)
 * @pre All accessors must be field accessor types
 * @pre IndexType must support SIMD operations
 *
 * @note All accessors must target the same medium type (enforced at
 * compile-time)
 * @note This function should be called from device kernels where thread-safety
 * is required
 * @note Atomic operations ensure correctness but may reduce performance
 * compared to add_on_device
 *
 * @warning Use atomic operations only when multiple threads may write to the
 * same memory location. Use regular add_on_device when thread-safety is not
 * required.
 *
 * Usage Examples:
 *
 * @code
 * // Elastic medium: Atomically add to displacement and velocity fields
 * auto disp = specfem::point::displacement<...>(...);
 * auto vel = specfem::point::velocity<...>(...);
 *
 * // Update field values in accessor
 * disp(0) += delta_disp_x;
 * disp(1) += delta_disp_z;
 *
 * // Single atomic kernel call updates multiple components thread-safely
 * atomic_add_on_device(simd_index, elastic_field, disp, vel);
 *
 * // Use in parallel assembly where race conditions may occur
 * Kokkos::parallel_for("assembly_kernel", range, KOKKOS_LAMBDA(int i) {
 *   // Multiple threads may write to same global indices
 *   atomic_add_on_device(global_index[i], field, accessor1, accessor2);
 * });
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename... AccessorTypes,
    typename std::enable_if_t<
        (specfem::data_access::is_field<AccessorTypes>::value && ...), int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
atomic_add_on_device(const IndexType &index, const ContainerType &field,
                     AccessorTypes &...accessors) {
  fields_impl::atomic_add_on_device(index, field, accessors...);
  return;
}
} // namespace specfem::assembly
