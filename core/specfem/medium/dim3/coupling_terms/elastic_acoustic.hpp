#pragma once

#include "specfem/algorithms.hpp"
#include "specfem/data_access.hpp"
#include "specfem/element.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/element_coupling.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::medium_physics::impl {

template <typename CoupledInterfaceType, typename CoupledFieldType,
          typename SelfFieldType>
KOKKOS_INLINE_FUNCTION void compute_coupling(
    const std::integral_constant<
        specfem::element::dimension_tag,
        specfem::element::dimension_tag::dim3> /*dimension_dispatch*/,
    const std::integral_constant<specfem::element_connections::type,
                                 specfem::element_connections::type::
                                     weakly_conforming> /*connection_dispatch*/,
    const std::integral_constant<specfem::element_coupling::interface_tag,
                                 specfem::element_coupling::interface_tag::
                                     elastic_acoustic> /*interface_dispatch*/,
    const CoupledInterfaceType &interface_data,
    const CoupledFieldType &coupled_field, SelfFieldType &self_field) {

  static_assert(specfem::data_access::is_acceleration<SelfFieldType>::value,
                "SelfFieldType must be an acceleration type");
  static_assert(specfem::data_access::is_acceleration<CoupledFieldType>::value,
                "CoupledFieldType must be an acceleration type");

  self_field(0) = interface_data.face_factor * interface_data.face_normal(0) *
                  coupled_field(0);
  self_field(1) = interface_data.face_factor * interface_data.face_normal(1) *
                  coupled_field(0);
  self_field(2) = interface_data.face_factor * interface_data.face_normal(2) *
                  coupled_field(0);
}
} // namespace specfem::medium_physics::impl
