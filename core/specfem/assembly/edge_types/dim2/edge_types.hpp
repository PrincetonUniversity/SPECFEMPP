#pragma once

#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/element_types.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <> class edge_types<specfem::dimension::type::dim2> {

private:
  using EdgeViewType =
      Kokkos::View<specfem::mesh_entity::edge *, Kokkos::DefaultExecutionSpace>;

public:
  constexpr static auto dimension = specfem::dimension::type::dim2;
  std::tuple<typename EdgeViewType::HostMirror,
             typename EdgeViewType::HostMirror>
  get_edges_on_host(const specfem::connections::type connection,
                    const specfem::interface::interface_tag edge,
                    const specfem::element::boundary_tag boundary) const;

  std::tuple<EdgeViewType, EdgeViewType>
  get_edges_on_device(const specfem::connections::type connection,
                      const specfem::interface::interface_tag edge,
                      const specfem::element::boundary_tag boundary) const;

  edge_types(
      const int ngllx, const int ngllz,
      const specfem::assembly::mesh<dimension> &mesh,
      const specfem::assembly::element_types<dimension> &element_types,
      const specfem::mesh::coupled_interfaces<dimension> &coupled_interfaces);

  edge_types() = default;

private:
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
                       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      DECLARE((EdgeViewType, self_edges),
                              (EdgeViewType::HostMirror, h_self_edges),
                              (EdgeViewType, coupled_edges),
                              (EdgeViewType::HostMirror, h_coupled_edges)))
};

} // namespace specfem::assembly
