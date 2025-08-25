#pragma once

#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "impl/utilities.hpp"
#include "kokkos_abstractions.h"
#include "mesh.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/jacobian.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

namespace {
using point = specfem::assembly::mesh_impl::dim2::utilities::point;
using bounding_box =
    specfem::assembly::mesh_impl::dim2::utilities::bounding_box;

std::tuple<Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>,
           int>
create_coordinate_arrays(const std::vector<point> &reordered_points, int nspec,
                         int ngll, int nglob) {

  // Create coordinate arrays (host-based since assign_numbering is host-only)
  auto index_mapping =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "index_mapping", nspec, ngll, ngll);
  auto coord =
      Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>(
          "coord", 2, nspec, ngll, ngll);

  std::vector<int> iglob_counted(nglob, -1);
  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;
  int iloc = 0;
  int inum = 0;

  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          if (iglob_counted[reordered_points[iloc].iglob] == -1) {
            const type_real x_cor = reordered_points[iloc].x;
            const type_real z_cor = reordered_points[iloc].z;

            iglob_counted[reordered_points[iloc].iglob] = inum;
            index_mapping(ispec, iz, ix) = inum;
            coord(0, ispec, iz, ix) = x_cor;
            coord(1, ispec, iz, ix) = z_cor;
            inum++;
          } else {
            index_mapping(ispec, iz, ix) =
                iglob_counted[reordered_points[iloc].iglob];
            coord(0, ispec, iz, ix) = reordered_points[iloc].x;
            coord(1, ispec, iz, ix) = reordered_points[iloc].z;
          }
          iloc++;
        }
      }
    }
  }

  int ngllxz = ngll * ngll;
  assert(nglob != (nspec * ngllxz));
  assert(inum == nglob);

  return std::make_tuple(index_mapping, coord, inum);
}

specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2>
assign_numbering(specfem::kokkos::HostView4d<double> global_coordinates) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxz = ngll * ngll;

  // Extract coordinates into testable utility functions
  auto points =
      specfem::assembly::mesh_impl::dim2::utilities::flatten_coordinates(
          global_coordinates);

  // Sort points spatially
  auto sorted_points = points;
  specfem::assembly::mesh_impl::dim2::utilities::sort_points_spatially(
      sorted_points);

  // Compute spatial tolerance
  type_real tolerance =
      specfem::assembly::mesh_impl::dim2::utilities::compute_spatial_tolerance(
          sorted_points, nspec, ngllxz);

  // Assign global numbering
  int nglob =
      specfem::assembly::mesh_impl::dim2::utilities::assign_global_numbering(
          sorted_points, tolerance);

  // Reorder points to original layout
  auto reordered_points =
      specfem::assembly::mesh_impl::dim2::utilities::reorder_to_original_layout(
          sorted_points);

  // Calculate bounding box using utilities
  auto bbox =
      specfem::assembly::mesh_impl::dim2::utilities::compute_bounding_box(
          reordered_points);

  // Create coordinate arrays
  auto [index_mapping, coord, nglob_actual] =
      create_coordinate_arrays(reordered_points, nspec, ngll, nglob);

  // Create mesh points object using constructor with pre-computed arrays
  specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2>
      mesh_points(nspec, ngll, ngll, nglob_actual, index_mapping, coord, bbox.xmin, bbox.xmax,
                  bbox.zmin, bbox.zmax);

  return mesh_points;
}

} // namespace

specfem::assembly::mesh<specfem::dimension::type::dim2>::mesh(
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
        &control_nodes_in,
    const specfem::quadrature::quadratures &quadratures,
    const specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>
        &adjacency_graph) {
  nspec = tags.nspec;
  ngllz = quadratures.gll.get_N();
  ngllx = quadratures.gll.get_N();
  ngnod = control_nodes_in.ngnod;

  auto &mapping =
      static_cast<specfem::assembly::mesh_impl::mesh_to_compute_mapping<
          specfem::dimension::type::dim2> &>(*this);
  auto &control_nodes = static_cast<specfem::assembly::mesh_impl::control_nodes<
      specfem::dimension::type::dim2> &>(*this);
  auto &quadrature = static_cast<specfem::assembly::mesh_impl::quadrature<
      specfem::dimension::type::dim2> &>(*this);
  auto &shape_functions =
      static_cast<specfem::assembly::mesh_impl::shape_functions<
          specfem::dimension::type::dim2> &>(*this);

  mapping = specfem::assembly::mesh_impl::mesh_to_compute_mapping<
      specfem::dimension::type::dim2>(tags);
  control_nodes = specfem::assembly::mesh_impl::control_nodes<
      specfem::dimension::type::dim2>(mapping, control_nodes_in);
  quadrature =
      specfem::assembly::mesh_impl::quadrature<specfem::dimension::type::dim2>(
          quadratures);

  shape_functions = specfem::assembly::mesh_impl::shape_functions<
      specfem::dimension::type::dim2>(
      quadratures.gll.get_hxi(), quadratures.gll.get_hxi(),
      quadratures.gll.get_N(), control_nodes_in.ngnod);

  if (adjacency_graph.empty()) {
    this->assemble();
  } else {
    this->assemble(adjacency_graph);
  }
}

void specfem::assembly::mesh<specfem::dimension::type::dim2>::assemble() {

  const int ngnod = this->ngnod;
  const int nspec = this->nspec;

  const int ngll = this->ngllx; // = ngllz

  const auto xi = this->h_xi;
  const auto gamma = this->h_xi;

  const auto shape2D = this->h_shape2D;
  const auto coord = this->h_control_node_coord;

  const int scratch_size =
      specfem::kokkos::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  specfem::kokkos::HostView4d<double> global_coordinates(
      "specfem::assembly::mesh::assemble::global_coordinates", nspec, ngll,
      ngll, 2);

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        auto shape_functions =
            Kokkos::subview(h_shape2D, iz, ix, Kokkos::ALL());

        double xcor = 0.0;
        double zcor = 0.0;

        for (int in = 0; in < ngnod; in++) {
          xcor += coord(0, ispec, in) * shape_functions[in];
          zcor += coord(1, ispec, in) * shape_functions[in];
        }

        global_coordinates(ispec, iz, ix, 0) = xcor;
        global_coordinates(ispec, iz, ix, 1) = zcor;
      }
    }
  }

  auto &points = static_cast<
      specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2> &>(
      *this);

  points = assign_numbering(global_coordinates);
}
