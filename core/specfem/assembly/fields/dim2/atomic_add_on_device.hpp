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
 * @brief Atomically adds field data on the device at the specified index from
 * the given accessors.
 *
 * @ingroup FieldDataAccess
 *
 * This function performs atomic element-wise addition of field data stored in
 * accessors to the field container at the specified index location. The atomic
 * operations ensure thread-safety when multiple threads may be writing to the
 * same memory location simultaneously, which is common in parallel assembly
 * operations.
 *
 * @tparam IndexType The type of the index (must be an index type with SIMD
 * support).
 * @tparam ContainerType The type of the container holding the field data.
 * @tparam AccessorTypes The types of the accessors used to access the field
 * data.
 *
 * @param index The index specifying the location or entity for atomic field
 * addition.
 * @param field The container holding the field data to be modified atomically.
 * @param accessors One or more accessor objects containing data to atomically
 * add to the field.
 *
 * @pre All accessors must have the same medium tag (e.g., all elastic or all
 * acoustic).
 * @pre All accessors must be field accessor types.
 * @pre IndexType must support SIMD operations.
 *
 * @note This function is device-only (KOKKOS_FORCEINLINE_FUNCTION) and should
 * be called from device kernels where thread-safety is required.
 *
 * @warning Use atomic operations only when necessary as they can reduce
 * performance. Use regular add_on_device when thread-safety is not required.
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
