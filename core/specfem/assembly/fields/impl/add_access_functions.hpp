#pragma once

#include "check_accessor_compatibility.hpp"
#include "field_impl.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly::fields_impl {

template <bool on_device, specfem::data_access::DataClassType DataClass,
          typename ContainerType, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
base_add_accessor(const int iglob, const int icomp, const ContainerType &field,
                  T &value) {

  using data_accessor =
      std::integral_constant<specfem::data_access::DataClassType, DataClass>;
  field.template get_value<on_device>(data_accessor(), iglob, icomp) += value;
}

template <bool on_device, specfem::data_access::DataClassType DataClass,
          typename MaskType, typename TagType, typename ContainerType,
          typename T>
KOKKOS_FORCEINLINE_FUNCTION void
base_add_accessor(const int iglob, const int icomp, const MaskType &mask,
                  const TagType tag_type, const ContainerType &field,
                  T &value) {

  using data_accessor =
      std::integral_constant<specfem::data_access::DataClassType, DataClass>;

  T lhs;
  Kokkos::Experimental::where(mask, lhs).copy_from(
      &(field.template get_value<on_device>(data_accessor(), iglob, icomp)),
      tag_type);
  lhs = lhs + value;
  Kokkos::Experimental::where(mask, lhs).copy_to(
      &(field.template get_value<on_device>(data_accessor(), iglob, icomp)),
      tag_type);
}

template <bool on_device, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_field_l<AccessorTypes>::value && ...),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void add_after_simd_dispatch(
    const std::false_type, const specfem::point::assembly_index<false> index,
    const ContainerType &field, AccessorTypes &...accessors) {

  check_accessor_compatibility<AccessorTypes...>();

  const int iglob = index.iglob;

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  constexpr static auto DataClass =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::data_class;

  // Call load for each accessor
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    (base_add_accessor<on_device, DataClass>(iglob, icomp, field,
                                             accessors(icomp)),
     ...);
  }
  return;
}

template <bool on_device, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_field_l<AccessorTypes>::value && ...),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void add_after_simd_dispatch(
    std::true_type, const specfem::point::assembly_index<true> index,
    const ContainerType &field, AccessorTypes &...accessors) {

  check_accessor_compatibility<AccessorTypes...>();

  using mask_type = typename std::tuple_element_t<
      0, std::tuple<AccessorTypes...> >::simd::mask_type;
  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const int iglob = index.iglob;

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  using TagType = typename std::tuple_element_t<
      0, std::tuple<AccessorTypes...> >::simd::tag_type;

  constexpr static auto DataClass =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::data_class;

  // Call load for each accessor
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    (base_add_accessor<on_device, DataClass>(iglob, icomp, mask, TagType(),
                                             field, accessors),
     ...);
  }
  return;
}

template <bool on_device, typename IndexType, typename ContainerType,
          typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_assembly_index<IndexType>::value &&
               (specfem::data_access::is_field_l<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
add_after_field_access(const IndexType &index, const ContainerType &field,
                       AccessorTypes &...accessors) {

  check_accessor_compatibility<AccessorTypes...>();

  using simd_accessor_type =
      std::integral_constant<bool, IndexType::using_simd>;

  // Call load for each accessor
  add_after_simd_dispatch<on_device>(simd_accessor_type(), index, field,
                                     accessors...);
  return;
}

} // namespace specfem::assembly::fields_impl
