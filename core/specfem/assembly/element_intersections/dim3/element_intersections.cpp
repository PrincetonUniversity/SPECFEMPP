#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>
#include <boost/graph/filtered_graph.hpp>

using FaceViewType = specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim3>::FaceViewType;

specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim3>::
    element_intersections(
        const int ngllx, const int nglly, const int ngllz,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types) {

  if (ngllz <= 0 || nglly <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  if (ngllz != ngllx || ngllz != nglly) {
    KOKKOS_ABORT_WITH_LOCATION(
        "The number of GLL points in z, y and x must be the same.");
  }

  const auto element =
      specfem::mesh_entity::element<dimension_tag>(ngllz, nglly, ngllx);

  const int ngll = ngllx; // ngllx == nglly == ngllz

  // Collect self/coupled face vectors for each tag combination
  struct CollectedFaces {
    std::vector<specfem::mesh_entity::face<dimension_tag> > self_collect;
    std::vector<specfem::mesh_entity::face<dimension_tag> > coupled_collect;
  };

  specfem::tag_dispatch::Storage<CollectedFaces, IntersectionCombinations>
      collected{ [&]<typename TagsType>() -> CollectedFaces {
        constexpr auto self_medium = specfem::element_coupling::attributes<
            TagsType::dimension_tag, TagsType::interface_tag>::self_medium();
        constexpr auto coupled_medium = specfem::element_coupling::attributes<
            TagsType::dimension_tag, TagsType::interface_tag>::coupled_medium();

        std::vector<specfem::mesh_entity::face<dimension_tag> > self_collect;
        std::vector<specfem::mesh_entity::face<dimension_tag> > coupled_collect;

        const auto &graph = mesh.graph();

        // Filter out corresponding connections
        auto filter = [&graph](const auto &edge) {
          return graph[edge].connection == TagsType::connection_tag;
        };

        // Create a filtered graph view
        const auto &nc_graph = boost::make_filtered_graph(graph, filter);
        int face_index = 0;
        for (const auto &edge :
             boost::make_iterator_range(boost::edges(nc_graph))) {
          const int ispec1 = boost::source(edge, nc_graph);
          const int ispec2 = boost::target(edge, nc_graph);
          const auto boundary_tag = element_types.get_boundary_tag(ispec1);
          const auto medium1 = element_types.get_medium_tag(ispec1);
          const auto medium2 = element_types.get_medium_tag(ispec2);
          if (boundary_tag == TagsType::boundary_tag &&
              medium1 == self_medium && medium2 == coupled_medium &&
              medium1 != medium2) {
            const specfem::mesh_entity::dim3::type self_orientation =
                nc_graph[edge].orientation;
            // Only process face connections (skip edge and corner connections)
            if (!specfem::mesh_entity::contains(
                    specfem::mesh_entity::dim3::faces, self_orientation)) {
              continue;
            }
            const auto [edge_inv, exists] =
                boost::edge(ispec2, ispec1, nc_graph);
            if (!exists) {
              throw std::runtime_error("Non-symmetric adjacency graph "
                                       "detected in `face_types`.");
            }
            const specfem::mesh_entity::dim3::type coupled_orientation =
                nc_graph[edge_inv].orientation;
            // we do not need orientation flipping -- that's handled by
            // the transfer function
            self_collect.push_back(
                { ispec1, self_orientation, face_index, false });
            coupled_collect.push_back(
                { ispec2, coupled_orientation, face_index, false });
            face_index++;
          }
        }
        return { std::move(self_collect), std::move(coupled_collect) };
      } };

  // Build host face views from collected faces
  h_self_faces = { [&]<typename TagsType>() -> FaceViewType::HostMirror {
    return face_view_from_collected_faces(
        "specfem::assembly::element_intersections::self_faces",
        collected.template get<TagsType>().self_collect, element);
  } };

  h_coupled_faces = { [&]<typename TagsType>() -> FaceViewType::HostMirror {
    return face_view_from_collected_faces(
        "specfem::assembly::element_intersections::coupled_faces",
        collected.template get<TagsType>().coupled_collect, element);
  } };

  // Allocate device views and deep-copy from host
  self_faces = specfem::tag_dispatch::create_mirror_storage_and_copy(
      Kokkos::DefaultExecutionSpace{}, h_self_faces);
  coupled_faces = specfem::tag_dispatch::create_mirror_storage_and_copy(
      Kokkos::DefaultExecutionSpace{}, h_coupled_faces);

  return;
}

std::tuple<FaceViewType::HostMirror, FaceViewType::HostMirror>
specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim3>::
    get_intersections_on_host(
        const specfem::element_connections::type connection,
        const specfem::element_coupling::interface_tag face,
        const specfem::element::boundary_tag boundary,
        const specfem::element_coupling::flux_scheme_tag flux_scheme) const {
  return std::make_tuple(
      h_self_faces.get(connection, face, boundary, flux_scheme),
      h_coupled_faces.get(connection, face, boundary, flux_scheme));
}

std::tuple<FaceViewType, FaceViewType> specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim3>::
    get_intersections_on_device(
        const specfem::element_connections::type connection,
        const specfem::element_coupling::interface_tag face,
        const specfem::element::boundary_tag boundary,
        const specfem::element_coupling::flux_scheme_tag flux_scheme) const {
  return std::make_tuple(
      self_faces.get(connection, face, boundary, flux_scheme),
      coupled_faces.get(connection, face, boundary, flux_scheme));
}

specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim3>::FaceViewType::HostMirror
specfem::assembly::face_view_from_collected_faces(
    const std::string &label,
    const std::vector<
        specfem::mesh_entity::face<specfem::element::dimension_tag::dim3> >
        &collected_faces,
    const specfem::mesh_entity::element<specfem::element::dimension_tag::dim3>
        &element) {
  const int ngll = element.ngll;
  const int count = collected_faces.size();

  specfem::assembly::element_intersections<
      specfem::element::dimension_tag::dim3>::FaceViewType::HostMirror
      face_view(label, count, ngll);

  for (int iface = 0; iface < count; iface++) {
    face_view.element_index(iface) = collected_faces[iface].ispec;
    face_view.face_index(iface) = collected_faces[iface].face_index;
    face_view.face_types(iface) = collected_faces[iface].face_type;
    for (int ipoint_i = 0; ipoint_i < ngll; ipoint_i++) {
      for (int ipoint_j = 0; ipoint_j < ngll; ipoint_j++) {
        // Linear index such that ipoint = point % ngll and jpoint = point /
        // ngll
        const auto [iz1, iy1, ix1] = element.map_coordinates(
            collected_faces[iface].face_type, ipoint_i + ipoint_j * ngll);
        face_view.iz(iface, ipoint_i, ipoint_j) = iz1;
        face_view.iy(iface, ipoint_i, ipoint_j) = iy1;
        face_view.ix(iface, ipoint_i, ipoint_j) = ix1;
      }
    }
  }
  return face_view;
}
