#pragma once

#include "check_accessor_compatibility.hpp"
#include "field_impl.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly::fields_impl {

template <bool on_device, typename ContainerType, typename AccessorType>
KOKKOS_FORCEINLINE_FUNCTION void
base_store_accessor(const int iglob, const int icomp,
                    const ContainerType &field, AccessorType &accessor) {

  static_assert(
      !AccessorType::simd::using_simd,
      "This function is only for non-SIMD accessors. Use the other overload.");

  constexpr static int ncomponents =
      specfem::element::attributes<AccessorType::dimension_tag,
                                   AccessorType::medium_tag>::components;

  using data_accessor =
      std::integral_constant<specfem::data_access::DataClassType,
                             AccessorType::data_class>;
  field.template get_value<on_device>(data_accessor(), iglob, icomp) =
      accessor(icomp);
}

template <bool on_device, typename MaskType, typename ContainerType,
          typename AccessorType>
KOKKOS_FORCEINLINE_FUNCTION void
base_store_accessor(const int iglob, const int icomp, const MaskType &mask,
                    const ContainerType &field, AccessorType &accessor) {

  static_assert(
      AccessorType::simd::using_simd,
      "This function is only for SIMD accessors. Use the other overload.");

  constexpr static int ncomponents =
      specfem::element::attributes<AccessorType::dimension_tag,
                                   AccessorType::medium_tag>::components;

  using data_accessor =
      std::integral_constant<specfem::data_access::DataClassType,
                             AccessorType::data_class>;

  using simd_type = typename AccessorType::simd::datatype;

  Kokkos::Experimental::where(mask, accessor(icomp))
      .copy_to(
          &(field.template get_value<on_device>(data_accessor(), iglob, icomp)),
          typename AccessorType::simd::tag_type());
}

template <bool on_device, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_field_l<AccessorTypes>::value && ...),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void store_after_simd_dispatch(
    const std::false_type, const specfem::point::assembly_index<false> index,
    const ContainerType &field, AccessorTypes &...accessors) {

  check_accessor_compatibility<AccessorTypes...>();

  const int iglob = index.iglob;

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  // Call load for each accessor
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    (base_store_accessor<on_device>(iglob, icomp, field, accessors), ...);
  }
  return;
}

template <bool on_device, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_field_l<AccessorTypes>::value && ...),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void store_after_simd_dispatch(
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

  // Call load for each accessor
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    (base_store_accessor<on_device>(iglob, icomp, mask, field, accessors), ...);
  }
  return;
}

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_assembly_index<IndexType>::value &&
               (specfem::data_access::is_field_l<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void store_on_device(const IndexType &index,
                                                 const ContainerType &field,
                                                 AccessorTypes &...accessors) {

  check_accessor_compatibility<AccessorTypes...>();

  using simd_accessor_type =
      std::integral_constant<bool, IndexType::using_simd>;

  // Call load for each accessor
  store_after_simd_dispatch<true>(simd_accessor_type(), index, field,
                                  accessors...);
  return;
}

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              ((specfem::data_access::is_index_type<IndexType>::value) &&
               (specfem::data_access::is_field_l<AccessorTypes>::value && ...)),
              int> = 0>
void store_on_host(const IndexType &index, const ContainerType &field,
                   AccessorTypes &...accessors) {

  check_accessor_compatibility<AccessorTypes...>();
  using simd_accessor_type =
      std::integral_constant<bool, IndexType::using_simd>;

  // Call load for each accessor
  store_after_simd_dispatch<false>(simd_accessor_type(), index, field,
                                   accessors...);
  return;
}

} // namespace specfem::assembly::fields_impl
