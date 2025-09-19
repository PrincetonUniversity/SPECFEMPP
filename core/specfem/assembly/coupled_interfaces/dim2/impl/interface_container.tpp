#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/coupled_interfaces.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "enumerations/macros.hpp"

template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
specfem::assembly::coupled_interfaces_impl::interface_container<
    specfem::dimension::type::dim2, InterfaceTag, BoundaryTag>::
    interface_container(
        const int ngllz, const int ngllx,
        const specfem::assembly::edge_types<specfem::dimension::type::dim2>
            &edge_types,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  if (ngllz <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  if (ngllz != ngllx) {
    KOKKOS_ABORT_WITH_LOCATION(
        "The number of GLL points in z and x must be the same.");
  }

  const auto connection_mapping =
      specfem::connections::connection_mapping(ngllx, ngllz);

  const auto [self_edges, coupled_edges] = edge_types.get_edges_on_host(
      specfem::connections::type::weakly_conforming, InterfaceTag, BoundaryTag);

  const auto nedges = self_edges.size();

  this->edge_factor = EdgeFactorView(
      "specfem::assembly::coupled_interfaces::edge_factor", nedges, ngllx);
  this->edge_normal = EdgeNormalView(
      "specfem::assembly::coupled_interfaces::edge_normal", nedges, ngllx, 2);

  this->h_edge_factor = Kokkos::create_mirror_view(edge_factor);
  this->h_edge_normal = Kokkos::create_mirror_view(edge_normal);

  const auto weights = mesh.h_weights;

  for (int i = 0; i < nedges; ++i) {
    const int ispec = self_edges(i).ispec;
    const auto edge_type = self_edges(i).edge_type;

    const int npoints =
        connection_mapping.number_of_points_on_orientation(edge_type);
    for (int ipoint = 0; ipoint < npoints; ++ipoint) {
      const auto [iz, ix] =
          connection_mapping.coordinates_at_edge(edge_type, ipoint);
      specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
                                      false>
          point_jacobian_matrix;
      specfem::point::index<specfem::dimension::type::dim2, false> point_index{
        ispec, iz, ix
      };
      specfem::assembly::load_on_host(point_index, jacobian_matrix,
                                      point_jacobian_matrix);
      const auto dn = point_jacobian_matrix.compute_normal(edge_type);
      this->h_edge_normal(i, ipoint, 0) = dn(0);
      this->h_edge_normal(i, ipoint, 1) = dn(1);
      const std::array<type_real, 2> w{ weights(ix), weights(iz) };
      this->h_edge_factor(i, ipoint) = [&]() {
        switch (edge_type) {
        case specfem::mesh_entity::type::bottom:
        case specfem::mesh_entity::type::top:
          return w[0];
        case specfem::mesh_entity::type::left:
        case specfem::mesh_entity::type::right:
          return w[1];
        default:
          KOKKOS_ABORT_WITH_LOCATION("Invalid edge type");
          return static_cast<type_real>(0.0);
        }
        return static_cast<type_real>(0.0);
      }();
    }
  }

  Kokkos::deep_copy(edge_factor, h_edge_factor);
  Kokkos::deep_copy(edge_normal, h_edge_normal);
}
