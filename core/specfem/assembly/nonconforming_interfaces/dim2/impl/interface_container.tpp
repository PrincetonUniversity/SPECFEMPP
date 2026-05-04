#pragma once

#include "compute_intersection.hpp"
#include "compute_intersection.tpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/nonconforming_interfaces.hpp"
#include "specfem/data_access.hpp"
#include "specfem/element_coupling/tags.hpp"
#include "specfem/enums.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/macros.hpp"

template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
specfem::assembly::nonconforming_interfaces_impl::interface_container<
    specfem::element::dimension_tag::dim2, InterfaceTag, BoundaryTag,
    specfem::element_connections::type::nonconforming, FluxSchemeTag>::
    interface_container(
        const int ngllz, const int ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim2> &element_intersections,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::element_coupling::flux_scheme_configuration
            &flux_scheme_config) {

  Kokkos::View<type_real *, Kokkos::HostSpace> interface_knots;
  Kokkos::View<type_real *, Kokkos::HostSpace> interface_weights;
  Kokkos::View<type_real **, Kokkos::HostSpace> interface_deriv;

  if (flux_scheme_config.was_interfacial_quadrature_set()) {
    const auto &interfacial_quadrature_rule =
        flux_scheme_config.get_interfacial_quadrature();
    interface_knots = interfacial_quadrature_rule.get_hxi();
    interface_weights = interfacial_quadrature_rule.get_hw();
    interface_deriv = interfacial_quadrature_rule.get_hhprime();
  } else {
    // if flux_scheme_config does not give a quadrature rule (e.g. default arg),
    // copy mesh quadrature
    interface_knots =
        Kokkos::View<type_real *, Kokkos::HostSpace>("interface_knots", ngllx);
    interface_weights = Kokkos::View<type_real *, Kokkos::HostSpace>(
        "interface_weights", ngllx);
    interface_deriv = Kokkos::View<type_real **, Kokkos::HostSpace>(
        "interface_deriv", ngllx, ngllx);
    for (int i = 0; i < mesh.h_xi.extent(0); i++) {
      interface_knots(i) = mesh.h_xi(i);
      interface_weights(i) = mesh.h_weights(i);
      for (int j = 0; j < mesh.h_xi.extent(0); j++) {
        interface_deriv(i, j) = mesh.h_hprime(i, j);
      }
    }
  }

  // TODO(nquad_intersection::runtime_vs_template): I forsee a bug where this
  // value, below, is set to something smaller than the kernel template
  // parameter NQuadIntersection, and the values in the chunk_edge data
  // container are not zero-padded. intersection/transfer View extents are set
  // by the runtime value. Keep this in mind! (This TODO message is copy-pasted
  // in other places -- remove all instances by TOOD(...) header when resolved)
  const int nquad_intersection = interface_knots.extent(0);

  if (ngllz <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  if (ngllz != ngllx) {
    KOKKOS_ABORT_WITH_LOCATION(
        "The number of GLL points in z and x must be the same.");
  }

  const auto element = specfem::mesh_entity::element(ngllz, ngllx);

  const auto [self_edges, coupled_edges] =
      element_intersections.get_intersections_on_host(
          specfem::element_connections::type::nonconforming, InterfaceTag,
          BoundaryTag, FluxSchemeTag);

  const auto N = self_edges.N;

  this->intersection_factor = EdgeFactorView(
      "specfem::assembly::nonconforming_interfaces::intersection_factor", N,
      nquad_intersection);

  this->intersection_normal = EdgeNormalView(
      "specfem::assembly::nonconforming_interfaces::intersection_normal", N,
      nquad_intersection, 2);
  this->h_intersection_normal = Kokkos::create_mirror_view(intersection_normal);

  // consider linking conjugate containers so that we don't need to do
  // set_transfer_functions twice.
  this->transfer_function = TransferFunctionView(
      "specfem::assembly::nonconforming_interfaces::transfer_function", N,
      nquad_intersection, ngllx);

  this->transfer_function_other = TransferFunctionView(
      "specfem::assembly::nonconforming_interfaces::transfer_function_other", N,
      nquad_intersection, ngllx);

  this->h_intersection_factor = Kokkos::create_mirror_view(intersection_factor);
  this->h_transfer_function = Kokkos::create_mirror_view(transfer_function);
  this->h_transfer_function_other =
      Kokkos::create_mirror_view(transfer_function_other);

  const auto weights = mesh.h_weights;

  // used when computing transfer functions
  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim2> *,
                     Kokkos::HostSpace>
      icoorg("icoorg", mesh.ngnod);
  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim2> *,
                     Kokkos::HostSpace>
      jcoorg("jcoorg", mesh.ngnod);

  for (int i = 0; i < N; ++i) {
    const int ispec = self_edges(i).element_index;
    const auto iedge_type = self_edges(i).edge_type;
    const int jspec = coupled_edges(i).element_index;
    const auto jedge_type = coupled_edges(i).edge_type;
    for (int inod = 0; inod < mesh.ngnod; inod++) {
      icoorg(inod).x = mesh.h_control_node_coord(0, ispec, inod);
      icoorg(inod).z = mesh.h_control_node_coord(1, ispec, inod);
      jcoorg(inod).x = mesh.h_control_node_coord(0, jspec, inod);
      jcoorg(inod).z = mesh.h_control_node_coord(1, jspec, inod);
    }
    auto transfer_subview =
        Kokkos::subview(h_transfer_function, i, Kokkos::ALL, Kokkos::ALL);
    auto transfer_subview_other =
        Kokkos::subview(h_transfer_function_other, i, Kokkos::ALL, Kokkos::ALL);
    specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
        icoorg, jcoorg, iedge_type, jedge_type, interface_knots, mesh.h_xi,
        transfer_subview, transfer_subview_other);
    // compute normal on edge
    const int npoints = element.number_of_points_on_orientation(iedge_type);

    // compute factor by finding first derivative of position
    // along the edge and multiplying by the quadrature weight
    const Kokkos::View<type_real **, Kokkos::HostSpace> dr_intersection(
        "dr_intersection", nquad_intersection, 2);

    for (int iquad = 0; iquad < nquad_intersection; iquad++) {
      dr_intersection(iquad, 0) = 0;
      dr_intersection(iquad, 1) = 0;
    }
    for (int iknot = 0; iknot < nquad_intersection; iknot++) {
      // get local coordinate (we can recover this from the transfer function by
      // interpolating x)
      type_real local_coord = 0;
      for (int ipoint = 0; ipoint < npoints; ipoint++) {
        local_coord += transfer_subview(iknot, ipoint) * mesh.h_xi(ipoint);
      }

      // get global coordinate -- we interpolate against shape prime
      const auto [xi, gamma] = [&]() -> std::pair<type_real, type_real> {
        if (iedge_type == specfem::mesh_entity::dim2::type::bottom) {
          return { local_coord, -1 };
        } else if (iedge_type == specfem::mesh_entity::dim2::type::right) {
          return { 1, local_coord };
        } else if (iedge_type == specfem::mesh_entity::dim2::type::top) {
          return { local_coord, 1 };
        } else {
          return { -1, local_coord };
        }
      }();
      const auto loc =
          jacobian::compute_locations(icoorg, mesh.ngnod, xi, gamma);

      // accumulate derivative at each quadrature point
      for (int iquad = 0; iquad < nquad_intersection; iquad++) {
        dr_intersection(iquad, 0) += interface_deriv(iquad, iknot) * loc.x;
        dr_intersection(iquad, 1) += interface_deriv(iquad, iknot) * loc.z;
      }
    }

    // convert dr to ds and multiply by weights
    for (int iquad = 0; iquad < nquad_intersection; iquad++) {
      this->h_intersection_factor(i, iquad) =
          interface_weights(iquad) *
          std::sqrt(dr_intersection(iquad, 0) * dr_intersection(iquad, 0) +
                    dr_intersection(iquad, 1) * dr_intersection(iquad, 1));

      type_real nx = dr_intersection(iquad, 1);
      type_real nz = -dr_intersection(iquad, 0);
      type_real mag = std::sqrt(nx * nx + nz * nz);

      // previous computation was a 90 deg clockwise rotation. it should be CCW
      // for these cases:
      if (iedge_type == specfem::mesh_entity::dim2::type::top ||
          iedge_type == specfem::mesh_entity::dim2::type::left) {
        mag *= -1;
      }
      this->h_intersection_normal(i, iquad, 0) = nx / mag;
      this->h_intersection_normal(i, iquad, 1) = nz / mag;
    }
  }

  Kokkos::deep_copy(intersection_factor, h_intersection_factor);
  Kokkos::deep_copy(intersection_normal, h_intersection_normal);

  Kokkos::deep_copy(transfer_function, h_transfer_function);
  Kokkos::deep_copy(transfer_function_other, h_transfer_function_other);
}
