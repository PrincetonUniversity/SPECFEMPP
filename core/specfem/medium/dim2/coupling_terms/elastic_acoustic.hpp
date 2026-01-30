#pragma once

#include "enumerations/interface.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/data_access.hpp"
#include "specfem/data_access/check_compatibility.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::medium::impl {

template <typename CoupledInterfaceType, typename CoupledFieldType,
          typename SelfFieldType>
KOKKOS_INLINE_FUNCTION void compute_coupling(
    const std::integral_constant<
        specfem::dimension::type,
        specfem::dimension::type::dim2> /*dimension_dispatch*/,
    const std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::weakly_conforming> /*connection_dispatch*/,
    const std::integral_constant<specfem::interface::interface_tag,
                                 specfem::interface::interface_tag::
                                     elastic_acoustic> /*interface_dispatch*/,
    const CoupledInterfaceType &interface_data,
    const CoupledFieldType &coupled_field, SelfFieldType &self_field) {

  static_assert(specfem::data_access::is_acceleration<SelfFieldType>::value,
                "SelfFieldType must be an acceleration type");
  static_assert(specfem::data_access::is_acceleration<CoupledFieldType>::value,
                "CoupledFieldType must be an acceleration type");

  self_field(0) = interface_data.edge_factor * interface_data.edge_normal(0) *
                  coupled_field(0);
  self_field(1) = interface_data.edge_factor * interface_data.edge_normal(1) *
                  coupled_field(0);
}

template <typename IndexType, typename TransferFunctionType,
          typename IntersectionNormalType, typename CoupledFieldType,
          typename IntersectionFieldViewType>
KOKKOS_INLINE_FUNCTION void compute_coupling(
    const std::integral_constant<
        specfem::dimension::type,
        specfem::dimension::type::dim2> /*dimension_dispatch*/,
    const std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::nonconforming> /*connection_dispatch*/,
    const std::integral_constant<specfem::interface::interface_tag,
                                 specfem::interface::interface_tag::
                                     elastic_acoustic> /*interface_dispatch*/,
    const IndexType &chunk_edge_index,
    const TransferFunctionType &transfer_function,
    const IntersectionNormalType &intersection_normal,
    const CoupledFieldType &coupled_field,
    IntersectionFieldViewType &intersection_field) {

  static_assert(
      specfem::data_access::is_chunk_edge<IndexType>::value,
      "The index for a nonconforming compute_coupling must be a chunk_edge.");

  static_assert(specfem::data_access::is_acceleration<CoupledFieldType>::value,
                "CoupledFieldType must be an acceleration type");

  specfem::algorithms::transfer(
      chunk_edge_index, transfer_function, coupled_field,
      [&](const auto &index, const auto &point) {
        const int iedge = index(0);
        const int iquad = index(1);
        intersection_field(iedge, iquad, 0) =
            intersection_normal(iedge, iquad, 0) * point(0);
        intersection_field(iedge, iquad, 1) =
            intersection_normal(iedge, iquad, 1) * point(0);
      });
}

} // namespace specfem::medium::impl
