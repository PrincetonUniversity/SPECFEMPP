#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include <Kokkos_Core.hpp>
#include <boost/graph/filtered_graph.hpp>

using EdgeViewType = specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim2>::EdgeViewType;

specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim2>::
    element_intersections(
        const int ngllx, const int ngllz,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types,
        const specfem::element_coupling::flux_scheme_configuration
            &flux_scheme_config) {

  if (ngllz <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  if (ngllz != ngllx) {
    KOKKOS_ABORT_WITH_LOCATION(
        "The number of GLL points in z and x must be the same.");
  }

  const auto element = specfem::mesh_entity::element(ngllz, ngllx);

  const int ngll = ngllx; // ngllx == ngllz in 2D

  const auto flux_scheme_tag = flux_scheme_config.get_flux_scheme_tag();

  // Count the number of interfaces for each combination of connection
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET),
       FLUX_SCHEME_TAG(NATURAL, SYMMETRIC_INTERIOR_PENALTY)),
      CAPTURE(h_self_edges, h_coupled_edges, self_edges, coupled_edges) {
        int count = 0;
        int edge_index = 0;
        constexpr auto self_medium = specfem::element_coupling::attributes<
            _dimension_tag_, _interface_tag_>::self_medium();
        constexpr auto coupled_medium = specfem::element_coupling::attributes<
            _dimension_tag_, _interface_tag_>::coupled_medium();

        std::vector<specfem::mesh_entity::edge<dimension_tag> > self_collect;
        std::vector<specfem::mesh_entity::edge<dimension_tag> > coupled_collect;

        const auto &graph = mesh.graph();

        // Filter edges by connection type
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
              medium2 == coupled_medium && medium1 != medium2 &&
              flux_scheme_tag == _flux_scheme_tag_) {
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
                    specfem::element_connections::type::weakly_conforming &&
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

        _h_self_edges_ = edge_view_from_collected_edges(
            "specfem::assembly::interface_types::self_edges_host_mirror",
            self_collect, element);
        _h_coupled_edges_ = edge_view_from_collected_edges(
            "specfem::assembly::interface_types::coupled_edges_host_mirror",
            coupled_collect, element);

        element_intersections::deep_copy(_self_edges_, _h_self_edges_);
        element_intersections::deep_copy(_coupled_edges_, _h_coupled_edges_);
      })

  return;
}

std::tuple<EdgeViewType::HostMirror, EdgeViewType::HostMirror>
specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim2>::
    get_intersections_on_host(
        const specfem::element_connections::type connection,
        const specfem::element_coupling::interface_tag edge,
        const specfem::element::boundary_tag boundary,
        const specfem::element_coupling::flux_scheme_tag flux_scheme) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET),
       FLUX_SCHEME_TAG(NATURAL, SYMMETRIC_INTERIOR_PENALTY)),
      CAPTURE(h_self_edges, h_coupled_edges) {
        if (_connection_tag_ == connection && _interface_tag_ == edge &&
            _boundary_tag_ == boundary && _flux_scheme_tag_ == flux_scheme) {
          return std::make_tuple(_h_self_edges_, _h_coupled_edges_);
        }
      })

  throw std::runtime_error(
      "Connection type, interface type or boundary type not found");
}

std::tuple<EdgeViewType, EdgeViewType> specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim2>::
    get_intersections_on_device(
        const specfem::element_connections::type connection,
        const specfem::element_coupling::interface_tag edge,
        const specfem::element::boundary_tag boundary,
        const specfem::element_coupling::flux_scheme_tag flux_scheme) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET),
       FLUX_SCHEME_TAG(NATURAL, SYMMETRIC_INTERIOR_PENALTY)),
      CAPTURE(self_edges, coupled_edges) {
        if (_connection_tag_ == connection && _interface_tag_ == edge &&
            _boundary_tag_ == boundary && _flux_scheme_tag_ == flux_scheme) {
          return std::make_tuple(_self_edges_, _coupled_edges_);
        }
      })

  throw std::runtime_error(
      "Connection type, interface type or boundary type not found");
}

specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim2>::EdgeViewType::HostMirror
specfem::assembly::edge_view_from_collected_edges(
    const std::string &label,
    const std::vector<
        specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2> >
        &self_collect,
    const specfem::mesh_entity::element<specfem::element::dimension_tag::dim2>
        &element) {
  const int &ngll = element.ngllx;
  const int &count = self_collect.size();

  specfem::assembly::element_intersections<
      specfem::element::dimension_tag::dim2>::EdgeViewType::HostMirror
      self_edges(label, count, ngll);
  for (int iedge = 0; iedge < count; iedge++) {
    self_edges.element_index(iedge) = self_collect[iedge].ispec;
    self_edges.edge_index(iedge) = self_collect[iedge].iedge;
    self_edges.edge_types(iedge) = self_collect[iedge].edge_type;
    for (int ipoint = 0; ipoint < ngll; ipoint++) {
      const auto [iz1, ix1] =
          element.map_coordinates(self_collect[iedge].edge_type, ipoint);
      self_edges.iz(iedge, ipoint) = iz1;
      self_edges.ix(iedge, ipoint) = ix1;
    }
  }
  return self_edges;
}
