#pragma once

#include "specfem/assembly/conforming_interfaces.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/element.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point.hpp"

template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
specfem::assembly::conforming_interfaces_impl::interface_container<
    specfem::element::dimension_tag::dim3, InterfaceTag, BoundaryTag,
    specfem::element_connections::type::weakly_conforming>::
    interface_container(
        const int ngllz, const int nglly, const int ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim3> &element_intersections,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  if (ngllz <= 0 || nglly <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  const auto [self_faces, coupled_faces] =
      element_intersections.get_intersections_on_host(
          specfem::element_connections::type::weakly_conforming, InterfaceTag,
          BoundaryTag, specfem::element_coupling::flux_scheme_tag::natural);

  const int N = self_faces.N;
  const int npoints = self_faces.n_points;

  this->face_factor =
      FaceFactorView("specfem::assembly::coupled_interfaces::face_factor", N,
                     npoints, npoints);
  this->face_normal =
      FaceNormalView("specfem::assembly::coupled_interfaces::face_normal", N,
                     npoints, npoints, 3);

  this->h_face_factor = Kokkos::create_mirror_view(face_factor);
  this->h_face_normal = Kokkos::create_mirror_view(face_normal);

  for (int i = 0; i < N; ++i) {
    const auto face = self_faces(i);
    for (int ipoint_i = 0; ipoint_i < npoints; ++ipoint_i) {
      for (int ipoint_j = 0; ipoint_j < npoints; ++ipoint_j) {
        const auto face_pt = face(ipoint_i, ipoint_j);

        specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                        true, false>
            point_jacobian_matrix;
        specfem::point::index<specfem::element::dimension_tag::dim3, false>
            point_index{ face_pt.ispec, face_pt.iz, face_pt.iy, face_pt.ix };
        specfem::assembly::load_on_host(point_index, jacobian_matrix,
                                        point_jacobian_matrix);

        const auto dn = point_jacobian_matrix.compute_normal(face_pt.face_type);
        this->h_face_normal(i, ipoint_i, ipoint_j, 0) = dn(0);
        this->h_face_normal(i, ipoint_i, ipoint_j, 1) = dn(1);
        this->h_face_normal(i, ipoint_i, ipoint_j, 2) = dn(2);

        this->h_face_factor(i, ipoint_i, ipoint_j) = [&]() {
          switch (face_pt.face_type) {
          case specfem::mesh_entity::dim3::type::left:
          case specfem::mesh_entity::dim3::type::right:
            // Face in (iy, iz) plane; integrate over iy and iz
            return mesh.h_weights(face_pt.iy) * mesh.h_weights(face_pt.iz);
          case specfem::mesh_entity::dim3::type::bottom:
          case specfem::mesh_entity::dim3::type::top:
            // Face in (ix, iy) plane; integrate over ix and iy
            return mesh.h_weights(face_pt.ix) * mesh.h_weights(face_pt.iy);
          case specfem::mesh_entity::dim3::type::front:
          case specfem::mesh_entity::dim3::type::back:
            // Face in (ix, iz) plane; integrate over ix and iz
            return mesh.h_weights(face_pt.ix) * mesh.h_weights(face_pt.iz);
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
}
