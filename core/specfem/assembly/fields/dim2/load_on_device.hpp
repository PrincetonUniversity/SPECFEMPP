#pragma once

#include "impl/load_access_functions.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
#include "specfem/assembly/fields/impl/load_access_functions.hpp"
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
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &field,
                                                AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  const auto &current_field = field.template get_field<MediumTag>();

  fields_impl::load_after_field_access<true>(index, current_field,
                                             accessors...);

  return;
}

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              ((specfem::data_access::is_index_type<IndexType>::value) &&
               (specfem::data_access::is_point<IndexType>::value ||
                specfem::data_access::is_chunk_element<IndexType>::value ||
                specfem::data_access::is_chunk_edge<IndexType>::value) &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &field,
                                                AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  using simd_accessor_type =
      std::integral_constant<bool, IndexType::using_simd>;
  simulation_field_impl::load_after_simd_dispatch<true>(
      simd_accessor_type(), index, field, accessors...);
  return;
}
} // namespace fields_impl

/**
 * @brief Loads field data on the device at the specified index into the given
 * accessors.
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType The type of the index (assembly index, point, or chunk
 * element).
 * @tparam ContainerType The type of the container holding the field data.
 * @tparam AccessorTypes The types of the accessors used to access the field
 * data.
 *
 * @param index The index specifying the location or entity for field access.
 * @param field The container holding the field data.
 * @param accessors One or more accessor objects specifying how to access the
 * field data.
 */
template <
    typename IndexType, typename ContainerType, typename... AccessorTypes,
    typename std::enable_if_t<
        (specfem::data_access::is_field<AccessorTypes>::value && ...), int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &field,
                                                AccessorTypes &...accessors) {
  fields_impl::load_on_device(index, field, accessors...);
  return;
}
} // namespace specfem::assembly
