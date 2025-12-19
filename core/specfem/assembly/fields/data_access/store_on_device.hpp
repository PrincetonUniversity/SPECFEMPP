#pragma once

#include "impl/store_access_functions.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
#include "specfem/assembly/fields/impl/store_access_functions.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

namespace fields_impl {

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              ((specfem::data_access::is_assembly_index<IndexType>::value) &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void store_on_device(const IndexType &index,
                                                 const ContainerType &field,
                                                 AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  const auto &current_field = field.template get_field<MediumTag>();

  fields_impl::store_after_field_access<true>(index, current_field,
                                              accessors...);

  return;
}
} // namespace fields_impl

/**
 * @brief High-performance device-side field data storage for spectral elements
 *
 * Public interface for storing field data from accessors into simulation fields
 * on GPU devices. This function provides a unified interface for writing values
 * from multiple field components simultaneously with optimized device memory
 * access patterns.
 *
 * @ingroup FieldsDataAccess
 *
 * @tparam IndexType Index type (specfem::point::assembly_index)
 * @tparam ContainerType Simulation field container (2D/3D specializations)
 * @tparam AccessorTypes Variadic field accessor types
 *
 * @param index Spatial index (element + quadrature point information)
 * @param field Simulation field container holding medium-specific field data to
 * be modified
 * @param accessors Variable number of field accessors containing data to store
 *
 * @pre All accessors must have the same medium tag (e.g., all elastic or all
 * acoustic)
 * @pre All accessors must be field accessor types
 *
 * @note All accessors must target the same medium type (enforced at
 * compile-time)
 * @note This function is device-only (KOKKOS_FORCEINLINE_FUNCTION) and should
 * be called from device kernels
 * @note For thread-safe operations where race conditions may occur, use
 * atomic_add_on_device instead
 *
 * Usage Examples:
 *
 * @code
 * // Elastic medium: Store computed displacement and velocity
 * auto disp = specfem::point::displacement<...>(...);
 * auto vel = specfem::point::velocity<...>(...);
 *
 * // Update accessor values
 * disp(0) = computed_disp_x;
 * disp(1) = computed_disp_z;
 * vel(0) = computed_vel_x;
 *
 * // Single kernel call stores multiple components to field
 * store_on_device(assembly_index, elastic_field, disp, vel);
 *
 * // Use in device kernels for field updates
 * Kokkos::parallel_for("store_kernel", range, KOKKOS_LAMBDA(int i) {
 *   // Compute new field values
 *   store_on_device(index[i], field, accessor1, accessor2);
 * });
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename... AccessorTypes,
    typename std::enable_if_t<
        (specfem::data_access::is_field<AccessorTypes>::value && ...), int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void store_on_device(const IndexType &index,
                                                 const ContainerType &field,
                                                 AccessorTypes &...accessors) {
  fields_impl::store_on_device(index, field, accessors...);
  return;
}

} // namespace specfem::assembly
