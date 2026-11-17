#include "unique_faces.hpp"
#include "specfem/point/global_coordinates.hpp"

/**
 * @brief Recovers the unique faces from a host-space FaceView
 */
specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::host_mirror_type
specfem::assembly::nonconforming_interfaces_impl::collect_unique_faces(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces,
    const int &ngllz, const int &nglly, const int &ngllx) {

  std::vector<specfem::mesh_entity::face<specfem::element::dimension_tag::dim3>>
      collected_faces;
  std::map<std::pair<int, specfem::mesh_entity::dim3::type>, int> face_indices;
  int num_self_faces = 0;

  // populate this enumeration
  for (int iface = 0; iface < self_faces.N; ++iface) {
    const auto key = std::make_pair<int, specfem::mesh_entity::dim3::type>(
        self_faces(iface).element_index, self_faces(iface).face_type);
    if (face_indices.find(key) == face_indices.end()) {
      // face_indices does not yet have this (ispec, self_face_type)
      face_indices[key] = num_self_faces;
      collected_faces.push_back(
          { key.first, key.second, num_self_faces, false });
      num_self_faces++;
    }
  }
  specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::host_mirror_type
      h_faces = specfem::assembly::face_view_from_collected_faces(
          "h_self_faces_on_intersection", collected_faces,
          specfem::mesh_entity::element<specfem::element::dimension_tag::dim3>(
              ngllz, nglly, ngllx));
  return h_faces;
}
specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::host_mirror_type
specfem::assembly::nonconforming_interfaces_impl::collect_unique_faces(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces) {
  return collect_unique_faces(self_faces, self_faces.n_points,
                              self_faces.n_points, self_faces.n_points);
}

specfem::assembly::nonconforming_interfaces_impl::FaceLengthTotal
specfem::assembly::nonconforming_interfaces_impl::face_length_sum(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  const int ngll = self_faces.n_points;
  const auto compute_characteristic_length = [&](const decltype(self_faces(
                                                 0)) &face) {
    const int nodes_per_element = mesh.ngnod;
    const auto &coordinates = mesh.h_control_node_coordinates;

    const auto get_coord = [&](const int &ipoint_i, const int &ipoint_j) {
      const auto face_pt = face(ipoint_i, ipoint_j);
      return specfem::point::global_coordinates<
          specfem::element::dimension_tag::dim3>(
          mesh.h_coord(face_pt.ispec, face_pt.iz, face_pt.iy, face_pt.ix, 0),
          mesh.h_coord(face_pt.ispec, face_pt.iz, face_pt.iy, face_pt.ix, 1),
          mesh.h_coord(face_pt.ispec, face_pt.iz, face_pt.iy, face_pt.ix, 2));
    };
    // largest diagonal on face
    return std::max(specfem::point::distance(get_coord(0, 0),
                                             get_coord(ngll - 1, ngll - 1)),
                    specfem::point::distance(get_coord(ngll - 1, 0),
                                             get_coord(0, ngll - 1)));
  };

  auto unique_faces =
      collect_unique_faces(self_faces, self_faces.n_points, self_faces.n_points,
                           self_faces.n_points);
  type_real charlen = 0;
  for (int iface = 0; iface < unique_faces.N; iface++) {
    charlen += compute_characteristic_length(unique_faces(iface));
  }
  return { charlen, static_cast<int>(unique_faces.N) };
}
