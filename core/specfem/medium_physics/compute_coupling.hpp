#pragma once

#include "specfem/chunk_edge/nonconforming_interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/data_access/accessor/chunk_edge.hpp"
#include "specfem/data_access/check_compatibility.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/enums.hpp"
#include "specfem/medium/dim2/coupling_terms/acoustic_elastic.hpp"
#include "specfem/medium/dim2/coupling_terms/elastic_acoustic.hpp"
#include "specfem/medium/dim3/coupling_terms/acoustic_elastic.hpp"
#include "specfem/medium/dim3/coupling_terms/elastic_acoustic.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::medium_physics {

/**
 * @brief Flux a coupled medium adds to the self medium at one interface point
 *
 * Evaluates, at a single quadrature point of the interface, the flux into the
 * self medium from the coupled medium's field, with \f$ f \f$ the integration
 * factor and \f$ \mathbf{n} \f$ the normal stored in `interface_data`:
 *
 * - `acoustic_elastic` (self acoustic, coupled elastic displacement
 *   \f$ \mathbf{u} \f$):
 *   \f$ \mathrm{flux} = f \, \mathbf{n} \cdot \mathbf{u} \f$
 * - `elastic_acoustic` (self elastic, coupled acoustic potential acceleration
 *   \f$ a \f$):
 *   \f$ \mathrm{flux}_d = f \, n_d \, a \f$
 *
 * The interface tag is an explicit template argument and is not read from
 * `InterfaceDataType`, so that data typed with the conjugate interface tag
 * (whose factor and normal are those of the host face) can be used with the
 * tag that selects the formula. `interface_data` may be any point type that
 * carries the factor and normal for the dimension (`edge_factor`/`edge_normal`
 * in 2D, `face_factor`/`face_normal` in 3D); its connection type is not
 * restricted.
 *
 * @tparam InterfaceTag Interface tag selecting the flux formula
 * @tparam InterfaceDataType Point data with factor and normal
 * @tparam CoupledFieldType Point field of the coupled medium of `InterfaceTag`
 * @tparam SelfFieldType Point field of the self medium of `InterfaceTag`
 *
 * @param interface_data Factor and normal at the quadrature point
 * @param coupled_field Field of the coupled medium at that point
 * @param flux Flux into the self medium (output)
 */
template <specfem::element_coupling::interface_tag InterfaceTag,
          typename InterfaceDataType, typename CoupledFieldType,
          typename SelfFieldType>
KOKKOS_INLINE_FUNCTION void
compute_interface_flux(const InterfaceDataType &interface_data,
                       const CoupledFieldType &coupled_field,
                       SelfFieldType &flux) {

  constexpr auto dimension_tag = InterfaceDataType::dimension_tag;

  static_assert(specfem::data_access::is_point<CoupledFieldType>::value &&
                    specfem::data_access::is_field<CoupledFieldType>::value,
                "coupled_field is not a point field type");
  static_assert(specfem::data_access::is_point<SelfFieldType>::value &&
                    specfem::data_access::is_field<SelfFieldType>::value,
                "flux is not a point field type");

  static_assert(
      specfem::element_coupling::attributes<dimension_tag,
                                            InterfaceTag>::self_medium() ==
          SelfFieldType::medium_tag,
      "Inconsistent self medium tag between interface tag and flux field");
  static_assert(
      specfem::element_coupling::attributes<dimension_tag,
                                            InterfaceTag>::coupled_medium() ==
          CoupledFieldType::medium_tag,
      "Inconsistent coupled medium tag between interface tag and coupled "
      "field");

  using dimension_dispatch =
      std::integral_constant<specfem::element::dimension_tag, dimension_tag>;
  using interface_dispatch =
      std::integral_constant<specfem::element_coupling::interface_tag,
                             InterfaceTag>;

  impl::compute_interface_flux(dimension_dispatch(), interface_dispatch(),
                               interface_data, coupled_field, flux);
}

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

  compute_interface_flux<interface_tag>(interface_data, coupled_field,
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
      specfem::data_access::is_codim1_chunk<IndexType>::value,
      "The index must be a chunk_edge (2d) or chunk_face (3d) index: the "
      "transfer function is coupled chunk -> self chunk.");
  static_assert(
      specfem::data_access::is_codim1_chunk<SelfFieldType>::value,
      "`self_field` must be a chunk_edge (2d) or chunk_face (3d) accessor.");
  static_assert(
      specfem::data_access::is_codim1_chunk<CoupledFieldType>::value,
      "coupled_field is not a chunk_edge (2d) or chunk_face (3d) accessor");

  static_assert(specfem::data_access::is_field<CoupledFieldType>::value,
                "coupled_field is not a field type");
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

  static_assert(
      specfem::data_access::is_packed_accessor<InterfaceDataType>::value,
      "The nonconforming interface data must be a packed accessor.");

  /* after the addition of `flux_scheme_tag` to the accessor template
   * parameters, accessor packs are no longer needed. However, since the 2d
   * implementation was made prior to that, this will stay in until the
   * `AccessorPack`s are fully deprecated.
   */
  impl::compute_coupling_expand(
      dimension_dispatch(), connection_dispatch(), interface_dispatch(),
      std::make_integer_sequence<std::size_t, InterfaceDataType::n_accessors>{},
      index, interface_data, coupled_field, self_field);
}

} // namespace specfem::medium_physics
