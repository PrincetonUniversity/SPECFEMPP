#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/data_access/accessor/chunk_edge.hpp"
#include "specfem/data_access/check_compatibility.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/medium/dim2/coupling_terms/acoustic_elastic.hpp"
#include "specfem/medium/dim2/coupling_terms/elastic_acoustic.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::medium_physics {

/**
 * @brief Computes coupling terms between different physical media
 *
 * Handles coupling interactions at interfaces between acoustic and elastic
 * media in spectral element simulations. Uses compile-time dispatch and
 * type validation to ensure consistent medium types.
 *
 * @tparam CoupledInterfaceType Interface data type (coupled interface)
 * @tparam CoupledFieldType Field type from coupled medium
 * @tparam SelfFieldType Field type from self medium (modified)
 *
 * @param interface_data Interface geometric data (factor, normal)
 * @param coupled_field Field data from coupled medium
 * @param self_field Field data from self medium (output)
 *
 * @code{.cpp}
 * specfem::medium_physics::compute_coupling(interface, coupled_field,
 * self_field);
 * @endcode
 */
template <
    typename CoupledInterfaceType, typename CoupledFieldType,
    typename SelfFieldType,
    std::enable_if_t<CoupledInterfaceType::connection_tag ==
                         specfem::element_connections::type::weakly_conforming,
                     int> = 0>
KOKKOS_INLINE_FUNCTION void
compute_coupling(const CoupledInterfaceType &interface_data,
                 const CoupledFieldType &coupled_field,
                 SelfFieldType &self_field) {

  static_assert(specfem::data_access::is_conforming_interface<
                    CoupledInterfaceType>::value,
                "interface_data is not a conforming interface type");
  constexpr auto dimension_tag = CoupledInterfaceType::dimension_tag;
  constexpr auto interface_tag = CoupledInterfaceType::interface_tag;
  constexpr auto connection_tag = CoupledInterfaceType::connection_tag;

  static_assert(specfem::data_access::is_point<CoupledFieldType>::value &&
                    specfem::data_access::is_field<CoupledFieldType>::value,
                "coupled_field is not a point field type");

  static_assert(specfem::data_access::is_field<SelfFieldType>::value,
                "self_field is not a field type");

  constexpr auto self_medium_tag = SelfFieldType::medium_tag;
  static_assert(
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::self_medium() ==
          self_medium_tag,
      "Inconsistent self medium tag between interface and self field");

  constexpr auto coupled_medium_tag = CoupledFieldType::medium_tag;

  static_assert(
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::coupled_medium() ==
          coupled_medium_tag,
      "Inconsistent coupled medium tag between interface and coupled field");

  using dimension_dispatch =
      std::integral_constant<specfem::element::dimension_tag, dimension_tag>;
  using connection_dispatch =
      std::integral_constant<specfem::element_connections::type,
                             connection_tag>;
  using interface_dispatch =
      std::integral_constant<specfem::element_coupling::interface_tag,
                             interface_tag>;

  impl::compute_coupling(dimension_dispatch(), connection_dispatch(),
                         interface_dispatch(), interface_data, coupled_field,
                         self_field);
}

namespace impl {

template <typename DimensionDispatch, typename ConnectionDispatch,
          typename InterfaceDispatch, typename IndexType,
          typename InterfaceDataType, typename CoupledFieldType,
          typename SelfFieldType, std::size_t... Is>
KOKKOS_INLINE_FUNCTION void compute_coupling_expand(
    DimensionDispatch dimension, ConnectionDispatch connection,
    InterfaceDispatch interface,
    const std::integer_sequence<std::size_t, Is...> /*unused*/,
    const IndexType &index, const InterfaceDataType &interface_data,
    const CoupledFieldType &coupled_field, SelfFieldType &self_field) {

  compute_coupling(dimension, connection, interface, index,
                   static_cast<const std::tuple_element_t<
                       Is, typename InterfaceDataType::packed_accessors> &>(
                       interface_data)...,
                   coupled_field, self_field);
}

} // namespace impl

/**
 * @brief Computes coupling terms between different physical media
 *
 * Handles coupling interactions at interfaces between acoustic and elastic
 * media in spectral element simulations. Uses compile-time dispatch and
 * type validation to ensure consistent medium types.
 *
 * @tparam CoupledInterfaceType Interface data type (coupled interface)
 * @tparam CoupledFieldType Field type from coupled medium
 * @tparam SelfFieldType Field type from self medium (modified)
 *
 * @param interface_data Interface geometric data (factor, normal)
 * @param coupled_field Field data from coupled medium
 * @param self_field Field data from self medium (output)
 *
 * @code{.cpp}
 * specfem::medium_physics::compute_coupling(interface, coupled_field,
 * self_field);
 * @endcode
 */
template <
    typename IndexType, typename InterfaceDataType, typename CoupledFieldType,
    typename SelfFieldType,
    std::enable_if_t<InterfaceDataType::connection_tag ==
                         specfem::element_connections::type::nonconforming,
                     int> = 0>
KOKKOS_INLINE_FUNCTION void compute_coupling(
    const IndexType &index, const InterfaceDataType &interface_data,
    const CoupledFieldType &coupled_field, SelfFieldType &self_field) {

  static_assert(specfem::data_access::is_nonconforming_interface<
                    InterfaceDataType>::value,
                "interface_data is not a nonconforming coupled interface type");
  constexpr auto dimension_tag = InterfaceDataType::dimension_tag;
  constexpr auto interface_tag = InterfaceDataType::interface_tag;
  constexpr auto connection_tag = InterfaceDataType::connection_tag;

  static_assert(
      specfem::data_access::is_chunk_edge<IndexType>::value,
      "The index for a nonconforming compute_coupling must be a chunk_edge.");

  static_assert(specfem::data_access::is_chunk_edge<CoupledFieldType>::value &&
                    specfem::data_access::is_field<CoupledFieldType>::value,
                "coupled_field is not a chunk_edge field type");
  constexpr auto coupled_medium_tag = CoupledFieldType::medium_tag;

  static_assert(
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::coupled_medium() ==
          coupled_medium_tag,
      "Inconsistent coupled medium tag between interface and coupled field");

  using dimension_dispatch =
      std::integral_constant<specfem::element::dimension_tag, dimension_tag>;
  using connection_dispatch =
      std::integral_constant<specfem::element_connections::type,
                             connection_tag>;
  using interface_dispatch =
      std::integral_constant<specfem::element_coupling::interface_tag,
                             interface_tag>;

  impl::compute_coupling_expand(
      dimension_dispatch(), connection_dispatch(), interface_dispatch(),
      std::make_integer_sequence<std::size_t, InterfaceDataType::n_accessors>{},
      index, interface_data, coupled_field, self_field);
}

} // namespace specfem::medium_physics
