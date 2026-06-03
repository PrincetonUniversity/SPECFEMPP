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
  const auto &weights = mesh.h_weights;

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
    const int ispec = self_faces(iface).element_index;
    const auto iface_type = self_faces(iface).face_type;
    const int jspec = coupled_faces(iface).element_index;
    const auto jface_type = coupled_faces(iface).face_type;

    // face local indices
    int ipoint_i, ipoint_j, ipoint_ortho;

    // max along local indices
    int nglli, ngllj;

    const auto [ix, iy, iz] = [&]() -> std::tuple<int &, int &, int &> {
      if (iface_type == specfem::mesh_entity::dim3::type::bottom) {
        ipoint_ortho = 0;
        nglli = ngllx;
        ngllj = nglly;
        return { ipoint_i, ipoint_j, ipoint_ortho };
      } else if (iface_type == specfem::mesh_entity::dim3::type::right) {
        ipoint_ortho = ngllx - 1;
        nglli = nglly;
        ngllj = ngllz;
        return { ipoint_ortho, ipoint_i, ipoint_j };
      } else if (iface_type == specfem::mesh_entity::dim3::type::top) {
        ipoint_ortho = ngllz - 1;
        nglli = ngllx;
        ngllj = nglly;
        return { ipoint_i, ipoint_j, ipoint_ortho };
      } else if (iface_type == specfem::mesh_entity::dim3::type::left) {
        ipoint_ortho = 0;
        nglli = nglly;
        ngllj = ngllz;
        return { ipoint_ortho, ipoint_i, ipoint_j };
      } else if (iface_type == specfem::mesh_entity::dim3::type::front) {
        ipoint_ortho = 0;
        nglli = ngllx;
        ngllj = ngllz;
        return { ipoint_i, ipoint_ortho, ipoint_j };
      } else { // back
        ipoint_ortho = nglly - 1;
        nglli = ngllx;
        ngllj = ngllz;
        return { ipoint_i, ipoint_ortho, ipoint_j };
      }
    }();

    for (int ipoint_i = 0; ipoint_i < nglli; ipoint_i++) {
      for (int ipoint_j = 0; ipoint_j < ngllj; ipoint_j++) {
        specfem::point::global_coordinates<dimension_tag> global_coord(
            Kokkos::subview(mesh.h_coord, ispec, iz, iy, ix, Kokkos::ALL));

        const auto [local_coords, found] =
            specfem::algorithms::locate_point_impl::locate_point(
                global_coord, mesh, jspec, jface_type);

        if (found) {
          this->h_coupled_coordinates(iface, ipoint_i, ipoint_j, 0) =
              local_coords.first;
          this->h_coupled_coordinates(iface, ipoint_i, ipoint_j, 1) =
              local_coords.second;
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
  Kokkos::deep_copy(face_factor, h_face_factor);
  Kokkos::deep_copy(face_normal, h_face_normal);
  Kokkos::deep_copy(coupled_coordinates, h_coupled_coordinates);
}
