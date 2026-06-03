#pragma once

// #include "compute_intersection.hpp"
// #include "compute_intersection.tpp"
#include "interface_container.hpp"

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

  const auto &N = self_faces.N;
  const auto &weights = mesh.h_weights;


  const int num_faces = 0; // TODO count them

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

  // // used when computing transfer functions
  // const Kokkos::View<specfem::point::global_coordinates<
  //                        specfem::element::dimension_tag::dim3> *,
  //                    Kokkos::HostSpace>
  //     icoorg("icoorg", mesh.ngnod);
  // const Kokkos::View<specfem::point::global_coordinates<
  //                        specfem::element::dimension_tag::dim3> *,
  //                    Kokkos::HostSpace>
  //     jcoorg("jcoorg", mesh.ngnod);

  // for (int i = 0; i < N; ++i) {
  //   const int ispec = self_edges(i).element_index;
  //   const auto iedge_type = self_edges(i).edge_type;
  //   const int jspec = coupled_edges(i).element_index;
  //   const auto jedge_type = coupled_edges(i).edge_type;
  //   for (int inod = 0; inod < mesh.ngnod; inod++) {
  //     icoorg(inod).x = mesh.h_control_node_coord(0, ispec, inod);
  //     icoorg(inod).z = mesh.h_control_node_coord(1, ispec, inod);
  //     jcoorg(inod).x = mesh.h_control_node_coord(0, jspec, inod);
  //     jcoorg(inod).z = mesh.h_control_node_coord(1, jspec, inod);
  //   }
  //   auto transfer_subview =
  //       Kokkos::subview(h_transfer_function, i, Kokkos::ALL, Kokkos::ALL);
  //   auto transfer_subview_other =
  //       Kokkos::subview(h_transfer_function_other, i, Kokkos::ALL,
  //       Kokkos::ALL);
  //   specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
  //       icoorg, jcoorg, iedge_type, jedge_type, interface_knots, mesh.h_xi,
  //       transfer_subview, transfer_subview_other);
  //   // compute normal on edge
  //   const int npoints = element.number_of_points_on_orientation(iedge_type);

  //   // compute factor by finding first derivative of position
  //   // along the edge and multiplying by the quadrature weight
  //   const Kokkos::View<type_real **, Kokkos::HostSpace> dr_intersection(
  //       "dr_intersection", nquad_intersection, 2);

  //   for (int iquad = 0; iquad < nquad_intersection; iquad++) {
  //     dr_intersection(iquad, 0) = 0;
  //     dr_intersection(iquad, 1) = 0;
  //   }
  //   for (int iknot = 0; iknot < nquad_intersection; iknot++) {
  //     // get local coordinate (we can recover this from the transfer function
  //     by
  //     // interpolating x)
  //     type_real local_coord = 0;
  //     for (int ipoint = 0; ipoint < npoints; ipoint++) {
  //       local_coord += transfer_subview(iknot, ipoint) * mesh.h_xi(ipoint);
  //     }

  //     // get global coordinate -- we interpolate against shape prime
  //     const auto [xi, gamma] = [&]() -> std::pair<type_real, type_real> {
  //       if (iedge_type == specfem::mesh_entity::dim3::type::bottom) {
  //         return { local_coord, -1 };
  //       } else if (iedge_type == specfem::mesh_entity::dim3::type::right) {
  //         return { 1, local_coord };
  //       } else if (iedge_type == specfem::mesh_entity::dim3::type::top) {
  //         return { local_coord, 1 };
  //       } else {
  //         return { -1, local_coord };
  //       }
  //     }();
  //     const auto loc =
  //         jacobian::compute_locations(icoorg, mesh.ngnod, xi, gamma);

  //     // accumulate derivative at each quadrature point
  //     for (int iquad = 0; iquad < nquad_intersection; iquad++) {
  //       dr_intersection(iquad, 0) += interface_deriv(iquad, iknot) * loc.x;
  //       dr_intersection(iquad, 1) += interface_deriv(iquad, iknot) * loc.z;
  //     }
  //   }

  //   // convert dr to ds and multiply by weights
  //   for (int iquad = 0; iquad < nquad_intersection; iquad++) {
  //     this->h_intersection_factor(i, iquad) =
  //         interface_weights(iquad) *
  //         std::sqrt(dr_intersection(iquad, 0) * dr_intersection(iquad, 0) +
  //                   dr_intersection(iquad, 1) * dr_intersection(iquad, 1));

  //     type_real nx = dr_intersection(iquad, 1);
  //     type_real nz = -dr_intersection(iquad, 0);
  //     type_real mag = std::sqrt(nx * nx + nz * nz);

  //     // previous computation was a 90 deg clockwise rotation. it should be
  //     CCW
  //     // for these cases:
  //     if (iedge_type == specfem::mesh_entity::dim3::type::top ||
  //         iedge_type == specfem::mesh_entity::dim3::type::left) {
  //       mag *= -1;
  //     }
  //     this->h_intersection_normal(i, iquad, 0) = nx / mag;
  //     this->h_intersection_normal(i, iquad, 1) = nz / mag;
  //   }
  // }
  Kokkos::deep_copy(face_factor, h_face_factor);
  Kokkos::deep_copy(face_normal, h_face_normal);
  Kokkos::deep_copy(coupled_coordinates, h_coupled_coordinates);
}
