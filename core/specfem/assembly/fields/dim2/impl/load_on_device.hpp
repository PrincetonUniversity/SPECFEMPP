#pragma once

#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
#include "specfem/assembly/fields/impl/field_impl.hpp"
#include "specfem/assembly/fields/impl/load_on_device.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly::simulation_field_impl {

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_index_type<IndexType>::value &&
               (specfem::data_access::is_field_l<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const std::false_type, const IndexType &index,
               const ContainerType &field, AccessorTypes &...accessors) {

  static_assert(
      ((specfem::data_access::CheckCompatibility<IndexType, ContainerType,
                                                 AccessorTypes>::value) &&
       ...),
      "Incompatible types in load_on_device");

  static_assert(!IndexType::using_simd,
                "IndexType must not use SIMD in this overload");

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  const auto current_field = field.template get_field<MediumTag>();

  const int iglob = field.template get_iglob<true>(index.ispec, index.iz,
                                                   index.ix, MediumTag);

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    (base_load_accessor(iglob, icomp, current_field, accessors), ...);
  }

  return;
}

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_index_type<IndexType>::value &&
               (specfem::data_access::is_field_l<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const std::true_type, const IndexType &index,
               const ContainerType &field, AccessorTypes &...accessors) {

  static_assert(IndexType::using_simd,
                "IndexType must use SIMD in this overload");

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  constexpr static int simd_size =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::simd::size();

  int iglob[simd_size];
  for (int lane = 0; lane < simd_size; ++lane) {
    iglob[lane] = index.mask(lane)
                      ? field.template get_iglob<true>(
                            index.ispec + lane, index.iz, index.ix, MediumTag)
                      : field.nglob + 1;
  }

  using mask_type = typename std::tuple_element_t<
      0, std::tuple<AccessorTypes...> >::simd::mask_type;
  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto current_field = field.template get_field<MediumTag>();

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  // Call load for each accessor
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    (base_load_accessor(iglob[icomp], icomp, mask, current_field, accessors),
     ...);
  }
}
} // namespace specfem::assembly::simulation_field_impl
