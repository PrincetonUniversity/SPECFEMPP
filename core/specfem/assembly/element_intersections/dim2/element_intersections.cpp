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

  // Collect self/coupled edge vectors for each tag combination
  struct CollectedEdges {
    std::vector<specfem::mesh_entity::edge<dimension_tag> > self_collect;
    std::vector<specfem::mesh_entity::edge<dimension_tag> > coupled_collect;
  };

  specfem::tag_dispatch::Storage<CollectedEdges, IntersectionCombinations>
      collected{ [&]<typename TagsType>() -> CollectedEdges {
        constexpr auto self_medium = specfem::element_coupling::attributes<
            TagsType::dimension_tag, TagsType::interface_tag>::self_medium();
        constexpr auto coupled_medium = specfem::element_coupling::attributes<
            TagsType::dimension_tag, TagsType::interface_tag>::coupled_medium();

        std::vector<specfem::mesh_entity::edge<dimension_tag> > self_collect;
        std::vector<specfem::mesh_entity::edge<dimension_tag> > coupled_collect;

        const auto &graph = mesh.graph();

        // Filter edges by connection type
        auto filter = [&graph](const auto &edge) {
          return graph[edge].connection == TagsType::connection_tag;
        };

        // Create a filtered graph view
        const auto &nc_graph = boost::make_filtered_graph(graph, filter);
        int edge_index = 0;
        for (const auto &edge :
             boost::make_iterator_range(boost::edges(nc_graph))) {
          const int ispec1 = boost::source(edge, nc_graph);
          const int ispec2 = boost::target(edge, nc_graph);
          const auto boundary_tag = element_types.get_boundary_tag(ispec1);
          const auto medium1 = element_types.get_medium_tag(ispec1);
          const auto medium2 = element_types.get_medium_tag(ispec2);

          if (boundary_tag == TagsType::boundary_tag &&
              medium1 == self_medium && medium2 == coupled_medium &&
              medium1 != medium2 &&
              flux_scheme_tag == TagsType::flux_scheme_tag) {
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
            if (TagsType::connection_tag ==
                    specfem::element_connections::type::weakly_conforming &&
                (specfem::mesh_entity::contains(
                     specfem::mesh_entity::dim2::corners, self_orientation) ||
                 specfem::mesh_entity::contains(
                     specfem::mesh_entity::dim2::corners,
                     coupled_orientation))) {
              // skip corner connections
              continue;
            }
            // we do not need orientation flipping -- that's handled by
            // the transfer function
            self_collect.push_back(
                { ispec1, edge_index, self_orientation, false });
            coupled_collect.push_back(
                { ispec2, edge_index, coupled_orientation, false });
            edge_index++;
          }
        }
        return { std::move(self_collect), std::move(coupled_collect) };
      } };

  // Build host edge views from collected edges
  h_self_edges = { [&]<typename TagsType>() -> EdgeViewType::HostMirror {
    return edge_view_from_collected_edges(
        "specfem::assembly::interface_types::self_edges",
        collected.template get<TagsType>().self_collect, element);
  } };

  h_coupled_edges = { [&]<typename TagsType>() -> EdgeViewType::HostMirror {
    return edge_view_from_collected_edges(
        "specfem::assembly::interface_types::coupled_edges",
        collected.template get<TagsType>().coupled_collect, element);
  } };

  // Allocate device views and deep-copy from host
  self_edges = specfem::tag_dispatch::create_mirror_storage_and_copy(
      Kokkos::DefaultExecutionSpace{}, h_self_edges);
  coupled_edges = specfem::tag_dispatch::create_mirror_storage_and_copy(
      Kokkos::DefaultExecutionSpace{}, h_coupled_edges);

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
  return std::make_tuple(
      h_self_edges.get(connection, edge, boundary, flux_scheme),
      h_coupled_edges.get(connection, edge, boundary, flux_scheme));
}

std::tuple<EdgeViewType, EdgeViewType> specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim2>::
    get_intersections_on_device(
        const specfem::element_connections::type connection,
        const specfem::element_coupling::interface_tag edge,
        const specfem::element::boundary_tag boundary,
        const specfem::element_coupling::flux_scheme_tag flux_scheme) const {
  return std::make_tuple(
      self_edges.get(connection, edge, boundary, flux_scheme),
      coupled_edges.get(connection, edge, boundary, flux_scheme));
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
