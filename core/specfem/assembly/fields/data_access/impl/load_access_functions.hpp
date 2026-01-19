#pragma once

#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
#include "specfem/assembly/fields/impl/field_impl.hpp"
#include "specfem/assembly/fields/impl/load_access_functions.hpp"
#include "specfem/data_access.hpp"
#include "specfem/execution.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly::simulation_field_impl {

template <bool on_device, typename IndexType, typename ContainerType,
          typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_index_type<IndexType>::value &&
               specfem::data_access::is_point<IndexType>::value &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_after_simd_dispatch(const std::false_type, const IndexType &index,
                         const ContainerType &field,
                         AccessorTypes &...accessors) {

  static_assert(!IndexType::using_simd,
                "IndexType must not use SIMD in this overload");

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  specfem::assembly::fields_impl::check_accessor_compatibility<
      AccessorTypes...>();

  const auto current_field = field.template get_field<MediumTag>();

  const int iglob = field.template get_iglob<on_device>(index, MediumTag);

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    (specfem::assembly::fields_impl::base_load_accessor<
         on_device, AccessorTypes::data_class>(iglob, icomp, current_field,
                                               accessors(icomp)),
     ...);
  }

  return;
}

template <bool on_device, typename IndexType, typename ContainerType,
          typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_index_type<IndexType>::value &&
               specfem::data_access::is_point<IndexType>::value &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_after_simd_dispatch(const std::true_type, const IndexType &index,
                         const ContainerType &field,
                         AccessorTypes &...accessors) {

  static_assert(IndexType::using_simd,
                "IndexType must use SIMD in this overload");

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  specfem::assembly::fields_impl::check_accessor_compatibility<
      AccessorTypes...>();

  constexpr static int simd_size =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::simd::size();

  int iglob[simd_size];
  for (int lane = 0; lane < simd_size; ++lane) {
    iglob[lane] =
        index.mask(lane)
            ? field.template get_iglob<on_device>(index, lane, MediumTag)
            : field.nglob + 1;
  }

  const auto &current_field = field.template get_field<MediumTag>();

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  // Call load for each accessor
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    (specfem::assembly::fields_impl::base_load_accessor<
         on_device, AccessorTypes::data_class>(
         iglob, icomp, [&](std::size_t lane) { return index.mask(lane); },
         current_field, accessors(icomp)),
     ...);
  }
}

template <bool on_device, typename IndexType, typename ContainerType,
          typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_chunk_element<IndexType>::value &&
               (specfem::data_access::is_index_type<IndexType>::value) &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_after_simd_dispatch(const std::false_type, const IndexType &index,
                         const ContainerType &field,
                         AccessorTypes &...accessors) {

  static_assert(!IndexType::using_simd,
                "IndexType must not use SIMD in this overload");

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  specfem::assembly::fields_impl::check_accessor_compatibility<
      AccessorTypes...>();

  const auto &current_field = field.template get_field<MediumTag>();

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  specfem::execution::for_each_level(
      index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &iterator_index) {
        const auto local_index = iterator_index.get_local_index();
        const auto point_index = iterator_index.get_index();

        const int iglob =
            field.template get_iglob<on_device>(point_index, MediumTag);

        for (int icomp = 0; icomp < ncomponents; ++icomp) {
          (specfem::assembly::fields_impl::base_load_accessor<
               on_device, AccessorTypes::data_class>(
               iglob, icomp, current_field, accessors(local_index, icomp)),
           ...);
        }
      });

  return;
}

template <bool on_device, typename IndexType, typename ContainerType,
          typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_chunk_element<IndexType>::value &&
               (specfem::data_access::is_index_type<IndexType>::value) &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_after_simd_dispatch(const std::true_type, const IndexType &index,
                         const ContainerType &field,
                         AccessorTypes &...accessors) {

  static_assert(IndexType::using_simd,
                "IndexType must use SIMD in this overload");

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  specfem::assembly::fields_impl::check_accessor_compatibility<
      AccessorTypes...>();

  constexpr static int simd_size =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::simd::size();

  const auto &current_field = field.template get_field<MediumTag>();

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  using simd_type = typename std::tuple_element_t<
      0, std::tuple<AccessorTypes...> >::simd::datatype;

  specfem::execution::for_each_level(
      index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &iterator_index) {
        const auto local_index = iterator_index.get_local_index();
        const auto point_index = iterator_index.get_index();

        int iglob[simd_size];
        for (int lane = 0; lane < simd_size; ++lane) {
          iglob[lane] = point_index.mask(lane)
                            ? field.template get_iglob<on_device>(
                                  point_index, lane, MediumTag)
                            : field.nglob + 1;
        }

        for (int icomp = 0; icomp < ncomponents; ++icomp) {
          (specfem::assembly::fields_impl::base_load_accessor<
               on_device, AccessorTypes::data_class>(
               iglob, icomp,
               [&](std::size_t lane) { return point_index.mask(lane); },
               current_field, accessors(local_index, icomp)),
           ...);
        }
      });

  return;
}

template <bool on_device, typename IndexType, typename ContainerType,
          typename... AccessorTypes,
          typename std::enable_if_t<
              (specfem::data_access::is_index_type<IndexType>::value &&
               specfem::data_access::is_chunk_edge<IndexType>::value &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_after_simd_dispatch(const std::false_type, const IndexType &index,
                         const ContainerType &field,
                         AccessorTypes &...accessors) {
  static_assert(!IndexType::using_simd,
                "IndexType must not use SIMD in this overload");
  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;
  // Check that all accessors have the same medium tag
  specfem::assembly::fields_impl::check_accessor_compatibility<
      AccessorTypes...>();
  const auto current_field = field.template get_field<MediumTag>();

  constexpr static int ncomponents = specfem::element::attributes<
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag,
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag>::
      components;

  specfem::execution::for_each_level(
      index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &iterator_index) {
        const auto point_index = iterator_index.get_local_index();
        const int iglob =
            field.template get_iglob<on_device>(point_index, MediumTag);
        for (int icomp = 0; icomp < ncomponents; ++icomp) {
          (specfem::assembly::fields_impl::base_load_accessor<
               on_device, AccessorTypes::data_class>(
               iglob, icomp, current_field,
               accessors(point_index.iedge, point_index.ipoint, icomp)),
           ...);
        }
      });

  return;
}

} // namespace specfem::assembly::simulation_field_impl
