#pragma once

#include "dim2/coupling_terms/acoustic_elastic.hpp"
#include "dim2/coupling_terms/elastic_acoustic.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::medium {

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
 * specfem::medium::compute_coupling(interface, coupled_field, self_field);
 * @endcode
 */
template <typename CoupledInterfaceType, typename CoupledFieldType,
          typename SelfFieldType>
KOKKOS_INLINE_FUNCTION void
compute_coupling(const CoupledInterfaceType &interface_data,
                 const CoupledFieldType &coupled_field,
                 SelfFieldType &self_field) {

  static_assert(
      specfem::data_access::is_coupled_interface<CoupledInterfaceType>::value,
      "interface_data is not a coupled interface type");
  static_assert(specfem::data_access::is_point<CoupledFieldType>::value &&
                    specfem::data_access::is_field<CoupledFieldType>::value,
                "coupled_field is not a point field type");
  static_assert(specfem::data_access::is_field<SelfFieldType>::value,
                "self_field is not a field type");

  constexpr auto dimension_tag = CoupledInterfaceType::dimension_tag;
  constexpr auto interface_tag = CoupledInterfaceType::interface_tag;
  constexpr auto connection_tag = CoupledInterfaceType::connection_tag;
  constexpr auto self_medium_tag = SelfFieldType::medium_tag;
  constexpr auto coupled_medium_tag = CoupledFieldType::medium_tag;

  static_assert(
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium() ==
          self_medium_tag,
      "Inconsistent self medium tag between interface and self field");
  static_assert(
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::coupled_medium() ==
          coupled_medium_tag,
      "Inconsistent coupled medium tag between interface and coupled field");

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type, dimension_tag>;
  using connection_dispatch =
      std::integral_constant<specfem::connections::type, connection_tag>;
  using interface_dispatch =
      std::integral_constant<specfem::interface::interface_tag, interface_tag>;

  impl::compute_coupling(dimension_dispatch(), connection_dispatch(),
                         interface_dispatch(), interface_data, coupled_field,
                         self_field);
}

} // namespace specfem::medium
