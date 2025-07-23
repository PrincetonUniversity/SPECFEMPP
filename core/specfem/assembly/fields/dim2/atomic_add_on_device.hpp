#pragma once

#include "impl/atomic_add_on_device.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/atomic_add_on_device.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              ((specfem::data_access::is_index_type<IndexType>::value) &&
               (specfem::data_access::is_field_l<AccessorTypes>::value && ...)),
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
  simulation_field_impl::atomic_add_on_device(simd_accessor_type(), index,
                                              field, accessors...);
  return;
}
} // namespace specfem::assembly
