#pragma once

#include "specfem/assembly/nonconforming_interfaces.hpp"
#include "specfem/data_access.hpp"

namespace specfem::assembly {

/**
 * @defgroup CoupledInterfaceDataAccess
 * @brief Data access functions for coupled interface computation data
 *
 */

/**
 * @brief Load interface data from container to point on host
 *
 * Loads coupled interface data using compile-time dispatch based on the point's
 * template parameters (connection, interface, and boundary types).
 *
 * @ingroup CoupledInterfaceDataAccess
 *
 * @tparam IndexType Edge index type
 * @tparam ContainerType Coupled interfaces container type
 * @tparam AccessorType Interface point (or edge) type
 *
 * @param index Edge index specifying the interface location
 * @param container Coupled interfaces container holding interface data
 * @param accessor Point or chunk_edge object where loaded data will be stored
 *
 * @pre index refers to valid mesh edge
 * @pre container is properly initialized
 * @pre point type matches supported interface combinations
 *
 * @note For host-side computations only. Use load_on_device for device code.
 */
template <typename IndexType, typename ContainerType, typename AccessorType,
          typename std::enable_if_t<
              ((specfem::data_access::is_edge_index<IndexType>::value) &&
               (specfem::data_access::is_nonconforming_interface<
                   ContainerType>::value)),
              int> = 0>
inline void load_on_host(const IndexType &index, const ContainerType &container,
                         AccessorType &accessor) {

  // static_assert(
  //     specfem::data_access::CheckCompatibility<IndexType, ContainerType,
  //                                              AccessorType>::value,
  //     "Incompatible types in load_on_host");

  using accessor_dispatch =
      std::integral_constant<specfem::datatype::AccessorType,
                             IndexType::accessor_type>;

  container
      .template get_interface_container<
          AccessorType::interface_tag, AccessorType::boundary_tag,
          AccessorType::connection_tag, AccessorType::flux_scheme_tag>()
      .template impl_load<false>(accessor_dispatch(), index, accessor);
}

/**
 * @brief Load interface data from container to edge on device
 *
 * Loads coupled interface data using compile-time dispatch based on the edge's
 * template parameters (connection, interface, and boundary types).
 *
 * @ingroup CoupledInterfaceDataAccess
 *
 * @tparam IndexType Edge index type
 * @tparam ContainerType Coupled interfaces container type
 * @tparam AccessorType Interface point (or edge) type
 *
 * @param index Edge index specifying the interface location
 * @param container Coupled interfaces container holding interface data
 * @param accessor Point or chunk_edge object where loaded data will be stored
 *
 * @pre index refers to valid mesh edge
 * @pre container is properly initialized
 * @pre point type matches supported interface combinations
 *
 * @note For device-side computations only. Use load_on_host for host code.
 */
template <typename IndexType, typename ContainerType, typename AccessorType,
          typename std::enable_if_t<
              ((specfem::data_access::is_edge_index<IndexType>::value) &&
               (specfem::data_access::is_nonconforming_interface<
                   ContainerType>::value)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &container,
                                                AccessorType &accessor) {
  // static_assert(
  //     specfem::data_access::CheckCompatibility<IndexType, ContainerType,
  //                                              AccessorType>::value,
  //     "Incompatible types in load_on_device");

  using accessor_dispatch =
      std::integral_constant<specfem::datatype::AccessorType,
                             IndexType::accessor_type>;
  container
      .template get_interface_container<
          AccessorType::interface_tag, AccessorType::boundary_tag,
          AccessorType::connection_tag, AccessorType::flux_scheme_tag>()
      .template impl_load<true>(accessor_dispatch(), index, accessor);

  return;
}
} // namespace specfem::assembly
