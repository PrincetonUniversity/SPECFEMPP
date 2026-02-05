#pragma once

#include "specfem/enums.hpp"
#include "specfem/element.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/assembly/conforming_interfaces.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/macros.hpp"

template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
specfem::assembly::conforming_interfaces_impl::interface_container<
    specfem::element::dimension_tag::dim2, InterfaceTag, BoundaryTag,
    specfem::element_connections::type::weakly_conforming>::
    interface_container(
        const int ngllz, const int ngllx,
        const specfem::assembly::edge_types<specfem::element::dimension_tag::dim2>
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

  const auto [self_edges, coupled_edges] = edge_types.get_edges_on_host(
      specfem::element_connections::type::weakly_conforming, InterfaceTag, BoundaryTag);

  const int nedges = self_edges.n_edges;
  const int npoints = self_edges.n_points;

  this->edge_factor = EdgeFactorView(
      "specfem::assembly::coupled_interfaces::edge_factor", nedges, ngllx);
  this->edge_normal = EdgeNormalView(
      "specfem::assembly::coupled_interfaces::edge_normal", nedges, ngllx, 2);

  this->h_edge_factor = Kokkos::create_mirror_view(edge_factor);
  this->h_edge_normal = Kokkos::create_mirror_view(edge_normal);

  const auto weights = mesh.h_weights;

  for (int i = 0; i < nedges; ++i) {
    const auto edge = self_edges(i);
    for (int ipoint = 0; ipoint < npoints; ++ipoint) {
      const auto edge_index = edge(ipoint);
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2, true,
                                      false>
          point_jacobian_matrix;
      specfem::point::index<specfem::element::dimension_tag::dim2, false> point_index{
        edge_index.ispec, edge_index.iz, edge_index.ix
      };
      specfem::assembly::load_on_host(point_index, jacobian_matrix,
                                      point_jacobian_matrix);
      const auto dn = point_jacobian_matrix.compute_normal(edge_index.edge_type);
      this->h_edge_normal(edge_index.iedge, edge_index.ipoint, 0) = dn(0);
      this->h_edge_normal(edge_index.iedge, edge_index.ipoint, 1) = dn(1);
      const std::array<type_real, 2> w{ weights(edge_index.ix),
                                        weights(edge_index.iz) };
      this->h_edge_factor(edge_index.iedge, edge_index.ipoint) = [&]() {
        switch (edge_index.edge_type) {
        case specfem::mesh_entity::dim2::type::bottom:
        case specfem::mesh_entity::dim2::type::top:
          return w[0];
        case specfem::mesh_entity::dim2::type::left:
        case specfem::mesh_entity::dim2::type::right:
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
