#pragma once

#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "specfem/jacobian.hpp"
#include "kokkos_abstractions.h"
#include "mesh.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem_setup.hpp"
#include "impl/utilities.tpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

namespace {
using point = specfem::assembly::mesh_impl::dim2::utilities::point;
using bounding_box = specfem::assembly::mesh_impl::dim2::utilities::bounding_box;

specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2>
create_mesh_points(
    const std::vector<point>& reordered_points,
    const bounding_box& bbox,
    int nspec, int ngll, int nglob) {

  specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2> mesh_points(
      nspec, ngll, ngll);

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
            mesh_points.h_index_mapping(ispec, iz, ix) = inum;
            mesh_points.h_coord(0, ispec, iz, ix) = x_cor;
            mesh_points.h_coord(1, ispec, iz, ix) = z_cor;
            inum++;
          } else {
            mesh_points.h_index_mapping(ispec, iz, ix) =
                iglob_counted[reordered_points[iloc].iglob];
            mesh_points.h_coord(0, ispec, iz, ix) = reordered_points[iloc].x;
            mesh_points.h_coord(1, ispec, iz, ix) = reordered_points[iloc].z;
          }
          iloc++;
        }
      }
    }
  }

  mesh_points.xmin = bbox.xmin;
  mesh_points.xmax = bbox.xmax;
  mesh_points.zmin = bbox.zmin;
  mesh_points.zmax = bbox.zmax;

  int ngllxz = ngll * ngll;
  assert(nglob != (nspec * ngllxz));
  assert(inum == nglob);

  Kokkos::deep_copy(mesh_points.index_mapping, mesh_points.h_index_mapping);
  Kokkos::deep_copy(mesh_points.coord, mesh_points.h_coord);

  return mesh_points;
}

specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2>
assign_numbering(specfem::kokkos::HostView4d<double> global_coordinates) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxz = ngll * ngll;

  // Extract coordinates into testable utility functions
  auto points = specfem::assembly::mesh_impl::dim2::utilities::flatten_coordinates(global_coordinates);

  // Sort points spatially
  auto sorted_points = points;
  specfem::assembly::mesh_impl::dim2::utilities::sort_points_spatially(sorted_points);

  // Compute spatial tolerance
  type_real tolerance = specfem::assembly::mesh_impl::dim2::utilities::compute_spatial_tolerance(sorted_points, nspec, ngllxz);

  // Assign global numbering
  int nglob = specfem::assembly::mesh_impl::dim2::utilities::assign_global_numbering(sorted_points, tolerance);

  // Reorder points to original layout
  auto reordered_points = specfem::assembly::mesh_impl::dim2::utilities::reorder_to_original_layout(sorted_points);

  // Calculate bounding box using utilities
  auto bbox = specfem::assembly::mesh_impl::dim2::utilities::compute_bounding_box(reordered_points);

  // Create and populate mesh points object
  return create_mesh_points(reordered_points, bbox, nspec, ngll, nglob);
}

} // namespace

specfem::assembly::mesh<specfem::dimension::type::dim2>::mesh(
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
        &control_nodes_in,
    const specfem::quadrature::quadratures &quadratures) {
  nspec = tags.nspec;
  ngllz = quadratures.gll.get_N();
  ngllx = quadratures.gll.get_N();
  ngnod = control_nodes_in.ngnod;

  auto &mapping = static_cast<specfem::assembly::mesh_impl::mesh_to_compute_mapping<
      specfem::dimension::type::dim2> &>(*this);
  auto &control_nodes = static_cast<
      specfem::assembly::mesh_impl::control_nodes<specfem::dimension::type::dim2> &>(
      *this);
  auto &quadrature = static_cast<
      specfem::assembly::mesh_impl::quadrature<specfem::dimension::type::dim2> &>(
      *this);
  auto &shape_functions = static_cast<specfem::assembly::mesh_impl::shape_functions<
      specfem::dimension::type::dim2> &>(*this);

  mapping = specfem::assembly::mesh_impl::mesh_to_compute_mapping<
      specfem::dimension::type::dim2>(tags);
  control_nodes =
      specfem::assembly::mesh_impl::control_nodes<specfem::dimension::type::dim2>(
          *static_cast<const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
              specfem::dimension::type::dim2> *>(this),
          control_nodes_in);
  quadrature =
      specfem::assembly::mesh_impl::quadrature<specfem::dimension::type::dim2>(
          quadratures);

  shape_functions =
      specfem::assembly::mesh_impl::shape_functions<specfem::dimension::type::dim2>(
          quadratures.gll.get_hxi(), quadratures.gll.get_hxi(),
          quadratures.gll.get_N(), control_nodes_in.ngnod);

  this->assemble();
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
        auto shape_functions = Kokkos::subview(h_shape2D, iz, ix, Kokkos::ALL());

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
      specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2> &>(*this);

  points = assign_numbering(global_coordinates);
}
