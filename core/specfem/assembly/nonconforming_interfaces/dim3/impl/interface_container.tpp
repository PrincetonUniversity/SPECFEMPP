#pragma once

#include "interface_container.hpp"
#include "specfem/algorithms/locate_point.hpp"
#include "specfem/point/global_coordinates.hpp"
#include <cmath>

template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
specfem::assembly::nonconforming_interfaces_impl::interface_container<
    specfem::element::dimension_tag::dim3, InterfaceTag, BoundaryTag,
    specfem::element_connections::type::nonconforming, FluxSchemeTag>::
    interface_container(
        const int &ngllz, const int &nglly, const int &ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim3> &element_intersections,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::element_coupling::flux_scheme_configuration
            &flux_scheme_config) {

  if (ngllz <= 0 || nglly <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  if (ngllz != ngllx || ngllz != nglly) {
    KOKKOS_ABORT_WITH_LOCATION(
        "The number of GLL points in z, y, and x must be the same.");
  }
  const int ngll = std::max(std::max(ngllz, nglly), ngllx);
  constexpr int ndim = specfem::element::dimension<dimension_tag>::dim;

  const auto [self_faces, coupled_faces] =
      element_intersections.get_intersections_on_host(
          specfem::element_connections::type::nonconforming, InterfaceTag,
          BoundaryTag, FluxSchemeTag);

  const auto &num_faces = self_faces.N;

  // for every node (ipoint, jpoint) of every self-face (ispec, self_face_type),
  // we want exactly one coupled face (coupled_faces[iface]) node to match. We
  // expect the intersection data for `iface` to use the `coupled_faces[iface]`
  // local coordinates.

  // here, enumerate the unique self-faces:
  //           face_indices[(ispec, self_face_type)] = self_face_index
  std::map<std::pair<int, specfem::mesh_entity::dim3::type>, int> face_indices;

  int num_self_faces = 0;

  // populate this enumeration
  for (int iface = 0; iface < num_faces; ++iface) {
    const auto key = std::make_pair<int, specfem::mesh_entity::dim3::type>(
        self_faces(iface).element_index, self_faces(iface).face_type);
    if (face_indices.find(key) == face_indices.end()) {
      // face_indices does not yet have this (ispec, self_face_type)
      face_indices[key] = num_self_faces;
      num_self_faces++;
    }
  }

  // eventually, `hit_face_index(self_face_index, ipoint, jpoint) == iface`
  // as we iterate the intersections, it will store the intersection index with
  // the closest point to the node ((ispec, self_face_type): ipoint, jpoint), or
  // -1 if no coupled face exists yet.
  Kokkos::View<int ***, Kokkos::HostSpace> hit_face_index(
      "hit_face_index", num_self_faces, ngll, ngll);
  Kokkos::deep_copy(hit_face_index, -1);

  // the distance from the closest point to the self-node found so far.
  Kokkos::View<type_real ***, Kokkos::HostSpace> hit_face_distance(
      "hit_face_distance", num_self_faces, ngll, ngll);

  face_factor =
      FaceFactorView("specfem::assembly::nonconforming_interfaces::face_factor",
                     num_faces, ngll, ngll);
  h_face_factor = Kokkos::create_mirror_view(face_factor);

  face_normal =
      FaceNormalView("specfem::assembly::nonconforming_interfaces::face_normal",
                     num_faces, ngll, ngll, ndim);
  h_face_normal = Kokkos::create_mirror_view(face_normal);

  coupled_coordinates = CoupledCoordinatesView(
      "specfem::assembly::nonconforming_interfaces::coupled_coordinates",
      num_faces, ngll, ngll, ndim - 1);
  h_coupled_coordinates = Kokkos::create_mirror_view(coupled_coordinates);

  for (int iface = 0; iface < num_faces; ++iface) {
    const auto &self_face = self_faces(iface);
    const auto &coupled_face = coupled_faces(iface);
    const int &ispec = self_face.element_index;
    const auto &iface_type = self_face.face_type;
    const int &jspec = coupled_face.element_index;
    const auto &jface_type = coupled_face.face_type;

    for (int ipoint_i = 0; ipoint_i < ngll; ipoint_i++) {
      for (int ipoint_j = 0; ipoint_j < ngll; ipoint_j++) {
        const auto self_face_pt = self_face(ipoint_i, ipoint_j);
        const int iz = self_face_pt.iz;
        const int iy = self_face_pt.iy;
        const int ix = self_face_pt.ix;
        specfem::point::global_coordinates<dimension_tag> global_coord(
            mesh.h_coord(ispec, iz, iy, ix, 0),
            mesh.h_coord(ispec, iz, iy, ix, 1),
            mesh.h_coord(ispec, iz, iy, ix, 2));

        const auto [local_coords, point_found] =
            specfem::algorithms::locate_point_impl::locate_point(
                global_coord, mesh, jspec, jface_type, false);

        // recover distance
        specfem::point::global_coordinates<dimension_tag> matched_point =
            specfem::algorithms::locate_point_impl::locate_point(
                local_coords, mesh, jspec, jface_type);
        const type_real dist =
            specfem::point::distance(global_coord, matched_point);

        const int self_face_index =
            face_indices[std::make_pair(ispec, iface_type)];

        bool use_this_coupling = true;
        if (hit_face_index(self_face_index, ipoint_i, ipoint_j) != -1) {
          // is this closer than the previous one?
          if (hit_face_distance(self_face_index, ipoint_i, ipoint_j) > dist) {
            // remove previous record
            const int iface_prior =
                hit_face_index(self_face_index, ipoint_i, ipoint_j);
            for (int idim = 0; idim < ndim - 1; idim++) {
              this->h_coupled_coordinates(iface_prior, ipoint_i, ipoint_j,
                                          idim) = NAN;
            }
          } else {
            // keep the old record
            use_this_coupling = false;
          }
        }

        if (use_this_coupling) {
          this->h_coupled_coordinates(iface, ipoint_i, ipoint_j, 0) =
              local_coords.first;
          this->h_coupled_coordinates(iface, ipoint_i, ipoint_j, 1) =
              local_coords.second;
          hit_face_index(self_face_index, ipoint_i, ipoint_j) = iface;
          hit_face_distance(self_face_index, ipoint_i, ipoint_j) = dist;
        } else {
          for (int idim = 0; idim < ndim - 1; idim++) {
            this->h_coupled_coordinates(iface, ipoint_i, ipoint_j, idim) = NAN;
          }
        }

        specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                        true, false>
            point_jacobian_matrix;
        specfem::point::index<specfem::element::dimension_tag::dim3, false>
            point_index{ ispec, iz, iy, ix };
        specfem::assembly::load_on_host(point_index, jacobian_matrix,
                                        point_jacobian_matrix);

        const auto dn = point_jacobian_matrix.compute_normal(iface_type);
        this->h_face_normal(iface, ipoint_i, ipoint_j, 0) = dn(0);
        this->h_face_normal(iface, ipoint_i, ipoint_j, 1) = dn(1);
        this->h_face_normal(iface, ipoint_i, ipoint_j, 2) = dn(2);
        this->h_face_factor(iface, ipoint_i, ipoint_j) = [&]() {
          switch (iface_type) {
          case specfem::mesh_entity::dim3::type::left:
          case specfem::mesh_entity::dim3::type::right:
            // Face in (iy, iz) plane; integrate over iy and iz
            return mesh.h_weights(iy) * mesh.h_weights(iz);
          case specfem::mesh_entity::dim3::type::bottom:
          case specfem::mesh_entity::dim3::type::top:
            // Face in (ix, iy) plane; integrate over ix and iy
            return mesh.h_weights(ix) * mesh.h_weights(iy);
          case specfem::mesh_entity::dim3::type::front:
          case specfem::mesh_entity::dim3::type::back:
            // Face in (ix, iz) plane; integrate over ix and iz
            return mesh.h_weights(ix) * mesh.h_weights(iz);
          default:
            KOKKOS_ABORT_WITH_LOCATION("Invalid face type");
            return static_cast<type_real>(0.0);
          }
        }();
      }
    }
  }

  specfem::mesh_entity::element<specfem::element::dimension_tag::dim3>
      element_ref(ngllz, nglly, ngllx);

  const auto compute_characteristic_length = [&](const int &ispec) {
    const int nodes_per_element = mesh.ngnod;
    const auto &coordinates = mesh.h_control_node_coordinates;
    type_real min_x = std::numeric_limits<type_real>::max();
    type_real max_x = std::numeric_limits<type_real>::lowest();
    type_real min_y = min_x, max_y = max_x;
    type_real min_z = min_x, max_z = max_x;

    for (int inode = 0; inode < nodes_per_element; ++inode) {
      const type_real x = coordinates(ispec, inode, 0);
      const type_real y = coordinates(ispec, inode, 1);
      const type_real z = coordinates(ispec, inode, 2);

      min_x = std::min(min_x, x);
      max_x = std::max(max_x, x);
      min_y = std::min(min_y, y);
      max_y = std::max(max_y, y);
      min_z = std::min(min_z, z);
      max_z = std::max(max_z, z);
    }

    return std::max({ max_x - min_x, max_y - min_y, max_z - min_z });
  };

  // ensure each node was properly hit
  for (const auto &[self_face_ispec_and_type, isf] : face_indices) {
    for (int ipoint = 0; ipoint < ngll; ++ipoint) {
      for (int jpoint = 0; jpoint < ngll; ++jpoint) {
        if (hit_face_index(isf, ipoint, jpoint) == -1) {
          // node not hit: error out. Unless the logic of "always accept the
          // first point",
          //            namely (    bool use_this_coupling = true;  ),
          // changes, this should never be entered.
          throw std::runtime_error("Failed to match coupled coordinate to face "
                                   "node: this face did not have a coupling!");
        } else {
          const int &ispec = self_face_ispec_and_type.first;
          const type_real charlen = compute_characteristic_length(ispec);
          // node hit: verify closeness
          if (hit_face_distance(isf, ipoint, jpoint) > 2e-2 * charlen) {
            const specfem::mesh_entity::dim3::type &iface_type =
                self_face_ispec_and_type.second;
            int iz, iy, ix;
            element_ref.get_face_coordinates(iface_type, ipoint, jpoint, iz, iy,
                                             ix);
            std::ostringstream oss;

            oss << "Exceedingly large distance between coupled coordinate "
                   "and face node\n"
                << "    (" << ipoint << ", " << jpoint << ")\n"
                << "on mesh_entity "
                << specfem::mesh_entity::dim3::to_string(iface_type)
                << " of element ispec = " << ispec << " (mesher element "
                << mesh.h_compute_to_mesh(ispec) + 1 << ")\n"
                << "    (ix = " << ix << ", iy = " << iy << ", iz = " << iz
                << ")\n";

            const specfem::point::global_coordinates<dimension_tag>
                global_coord_selfnode(mesh.h_coord(ispec, iz, iy, ix, 0),
                                      mesh.h_coord(ispec, iz, iy, ix, 1),
                                      mesh.h_coord(ispec, iz, iy, ix, 2));
            oss << "    (x = " << global_coord_selfnode.x
                << ", y = " << global_coord_selfnode.y
                << ", z = " << global_coord_selfnode.z << ")\n"
                << "Smallest distance found: "
                << hit_face_distance(isf, ipoint, jpoint)
                << "\n                        (" << std::scientific
                << hit_face_distance(isf, ipoint, jpoint) / charlen
                << " ⨉ characteristic length)\n";

            const int &intersection_index = hit_face_index(isf, ipoint, jpoint);
            const auto &coupled_face = coupled_faces(intersection_index);
            const int &jspec = coupled_face.element_index;
            const auto &jface_type = coupled_face.face_type;
            const auto &local_coupled_coords = std::make_pair(
                h_coupled_coordinates(intersection_index, ipoint, jpoint, 0),
                h_coupled_coordinates(intersection_index, ipoint, jpoint, 1));
            specfem::point::global_coordinates<dimension_tag> matched_point =
                specfem::algorithms::locate_point_impl::locate_point(
                    local_coupled_coords, mesh, jspec, jface_type);

            oss << "Coupled Face: ispec = " << jspec << " (mesher element "
                << mesh.h_compute_to_mesh(jspec) + 1 << "), face = "
                << specfem::mesh_entity::dim3::to_string(jface_type)
                << "\n    (x = " << matched_point.x
                << ", y = " << matched_point.y << ", z = " << matched_point.z
                << ")\n"
                << "\n    local: (" << local_coupled_coords.first << ", "
                << local_coupled_coords.second << ")\n"
                << "on mesh_entity ";

            throw std::runtime_error(oss.str());
          }
        }
      }
    }
  }

  Kokkos::deep_copy(face_factor, h_face_factor);
  Kokkos::deep_copy(face_normal, h_face_normal);
  Kokkos::deep_copy(coupled_coordinates, h_coupled_coordinates);
}
