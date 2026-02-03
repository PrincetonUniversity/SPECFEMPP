#include "specfem/assembly/edge_types.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>
#include <boost/graph/filtered_graph.hpp>

using EdgeViewType =
    specfem::assembly::edge_types<specfem::dimension::type::dim2>::EdgeViewType;

specfem::assembly::edge_types<specfem::dimension::type::dim2>::edge_types(
    const int ngllx, const int ngllz,
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::element_types<dimension_tag> &element_types) {

  if (ngllz <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  if (ngllz != ngllx) {
    KOKKOS_ABORT_WITH_LOCATION(
        "The number of GLL points in z and x must be the same.");
  }

  const auto element = specfem::mesh_entity::element(ngllz, ngllx);

  const int ngll = ngllx; // ngllx == ngllz in 2D

  // Count the number of interfaces for each combination of connection
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(h_self_edges, h_coupled_edges, self_edges, coupled_edges) {
        int count = 0;
        int edge_index = 0;
        constexpr auto self_medium =
            specfem::interface::attributes<_dimension_tag_,
                                           _interface_tag_>::self_medium();
        constexpr auto coupled_medium =
            specfem::interface::attributes<_dimension_tag_,
                                           _interface_tag_>::coupled_medium();

        std::vector<specfem::mesh_entity::edge<dimension_tag> > self_collect;
        std::vector<specfem::mesh_entity::edge<dimension_tag> > coupled_collect;

        const auto &graph = mesh.graph();

        // Filter out corresponding connections
        auto filter = [&graph](const auto &edge) {
          return graph[edge].connection == _connection_tag_;
        };

        // Create a filtered graph view
        const auto &nc_graph = boost::make_filtered_graph(graph, filter);
        for (const auto &edge :
             boost::make_iterator_range(boost::edges(nc_graph))) {
          const int ispec1 = boost::source(edge, nc_graph);
          const int ispec2 = boost::target(edge, nc_graph);
          const auto boundary_tag = element_types.get_boundary_tag(ispec1);
          const auto medium1 = element_types.get_medium_tag(ispec1);
          const auto medium2 = element_types.get_medium_tag(ispec2);
          if (boundary_tag == _boundary_tag_ && medium1 == self_medium &&
              medium2 == coupled_medium && medium1 != medium2) {
            const specfem::mesh_entity::dim2::type self_orientation =
                nc_graph[edge].orientation;
            const auto [edge_inv, exists] =
                boost::edge(ispec2, ispec1, nc_graph);
            if (!exists) {
              throw std::runtime_error("Non-symmetric adjacency graph "
                                       "detected in `compute_intersection`.");
            }
            const specfem::mesh_entity::dim2::type coupled_orientation =
                nc_graph[edge_inv].orientation;
            if (_connection_tag_ ==
                    specfem::connections::type::weakly_conforming &&
                (specfem::mesh_entity::contains(
                     specfem::mesh_entity::dim2::corners, self_orientation) ||
                 specfem::mesh_entity::contains(
                     specfem::mesh_entity::dim2::corners,
                     coupled_orientation))) {
              // skip corner connections
              continue;
            }
            count++;
            // we do not need orientation flipping -- that's handled by
            // the transfer function
            self_collect.push_back(
                { ispec1, edge_index, self_orientation, false });
            coupled_collect.push_back(
                { ispec2, edge_index, coupled_orientation, false });
            edge_index++;
          }
        }

        _self_edges_ = EdgeViewType(
            "specfem::assembly::interface_types::self_edges", count, ngll);
        _coupled_edges_ = EdgeViewType(
            "specfem::assembly::interface_types::coupled_edges", count, ngll);
        _h_self_edges_ = edge_types::create_mirror_view(_self_edges_);
        _h_coupled_edges_ = edge_types::create_mirror_view(_coupled_edges_);

        for (int iedge = 0; iedge < count; iedge++) {
          _h_self_edges_.element_index(iedge) = self_collect[iedge].ispec;
          _h_self_edges_.edge_index(iedge) = self_collect[iedge].iedge;
          _h_self_edges_.edge_types(iedge) = self_collect[iedge].edge_type;
          _h_coupled_edges_.element_index(iedge) = coupled_collect[iedge].ispec;
          _h_coupled_edges_.edge_index(iedge) = coupled_collect[iedge].iedge;
          _h_coupled_edges_.edge_types(iedge) =
              coupled_collect[iedge].edge_type;
          for (int ipoint = 0; ipoint < ngll; ipoint++) {
            const auto [iz1, ix1] =
                element.map_coordinates(self_collect[iedge].edge_type, ipoint);
            _h_self_edges_.iz(iedge, ipoint) = iz1;
            _h_self_edges_.ix(iedge, ipoint) = ix1;
            const auto [iz2, ix2] = element.map_coordinates(
                coupled_collect[iedge].edge_type, ipoint);
            _h_coupled_edges_.iz(iedge, ipoint) = iz2;
            _h_coupled_edges_.ix(iedge, ipoint) = ix2;
          }
        }

        edge_types::deep_copy(_self_edges_, _h_self_edges_);
        edge_types::deep_copy(_coupled_edges_, _h_coupled_edges_);
      })

  return;
}

std::tuple<EdgeViewType::HostMirror, EdgeViewType::HostMirror>
specfem::assembly::edge_types<specfem::dimension::type::dim2>::
    get_edges_on_host(const specfem::connections::type connection,
                      const specfem::interface::interface_tag edge,
                      const specfem::element::boundary_tag boundary) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(h_self_edges, h_coupled_edges) {
        if (_connection_tag_ == connection && _interface_tag_ == edge &&
            _boundary_tag_ == boundary) {
          return std::make_tuple(_h_self_edges_, _h_coupled_edges_);
        }
      })

  throw std::runtime_error(
      "Connection type, interface type or boundary type not found");
}

std::tuple<EdgeViewType, EdgeViewType>
specfem::assembly::edge_types<specfem::dimension::type::dim2>::
    get_edges_on_device(const specfem::connections::type connection,
                        const specfem::interface::interface_tag edge,
                        const specfem::element::boundary_tag boundary) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(self_edges, coupled_edges) {
        if (_connection_tag_ == connection && _interface_tag_ == edge &&
            _boundary_tag_ == boundary) {
          return std::make_tuple(_self_edges_, _coupled_edges_);
        }
      })

  throw std::runtime_error(
      "Connection type, interface type or boundary type not found");
}
