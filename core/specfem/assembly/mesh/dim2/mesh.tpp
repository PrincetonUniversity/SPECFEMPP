#pragma once

#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "kokkos_abstractions.h"
#include "mesh.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/shape_functions.hpp"
#include "impl/utilities.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

namespace {
using point = specfem::assembly::mesh_impl::dim2::point;
using bounding_box = specfem::assembly::mesh_impl::dim2::bounding_box;

specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2>
assign_numbering(specfem::kokkos::HostView4d<double> global_coordinates) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxz = ngll * ngll;

  // Extract coordinates into testable utility functions
  auto points = specfem::assembly::mesh_impl::dim2::flatten_coordinates(
      global_coordinates);

  // Sort points spatially
  auto sorted_points = points;
  specfem::assembly::mesh_impl::dim2::sort_points_spatially(sorted_points);

  // Compute spatial tolerance
  type_real tolerance =
      specfem::assembly::mesh_impl::dim2::compute_spatial_tolerance(
          sorted_points, nspec, ngllxz);

  // Assign global numbering
  int nglob = specfem::assembly::mesh_impl::dim2::assign_global_numbering(
      sorted_points, tolerance);

  // Reorder points to original layout
  auto reordered_points =
      specfem::assembly::mesh_impl::dim2::reorder_to_original_layout(
          sorted_points);

  // Calculate bounding box using utilities
  auto bbox = specfem::assembly::mesh_impl::dim2::compute_bounding_box(
      reordered_points);

  // Create coordinate arrays
  auto [index_mapping, coord, nglob_actual] =
      specfem::assembly::mesh_impl::dim2::create_coordinate_arrays(
          reordered_points, nspec, ngll, nglob);

  // Create mesh points object using constructor with pre-computed arrays
  specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2>
      mesh_points(nspec, ngll, ngll, nglob_actual, index_mapping, coord,
                  bbox.xmin, bbox.xmax, bbox.zmin, bbox.zmax);
  return mesh_points;
}

// We need to build a new graph since the element numbering may have changed
// after the mesh assembly
specfem::assembly::mesh_impl::adjacency_graph<specfem::dimension::type::dim2>
build_assembly_adjacency_graph(
    const int nspec,
    const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
        specfem::dimension::type::dim2> &mapping,
    const specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>
        &mesh_adjacency_graph) {

  if (mesh_adjacency_graph.empty()) {
    return specfem::assembly::mesh_impl::adjacency_graph<
        specfem::dimension::type::dim2>();
  }

  specfem::assembly::mesh_impl::adjacency_graph<specfem::dimension::type::dim2>
      adjacency_graph(nspec);

  auto &g = adjacency_graph.graph();
  const auto &mesh_g = mesh_adjacency_graph.graph();

  for (int ispec = 0; ispec < nspec; ispec++) {
    // Get mesh index
    const int ispec_mesh = mapping.compute_to_mesh(ispec);
    // Iterate over all outgoing edges
    for (auto iedge :
         boost::make_iterator_range(boost::out_edges(ispec_mesh, mesh_g))) {
      // Get the target mesh index
      const int target_ispec_mesh = boost::target(iedge, mesh_g);
      // Get the target specfem index
      const int target_ispec = mapping.mesh_to_compute(target_ispec_mesh);
      // Get edge property
      const auto edge_property = mesh_g[iedge];
      // Add the edge to the adjacency graph
      boost::add_edge(ispec, target_ispec, edge_property, g);
    }
  }

  // Check that the graph is symmetric
  adjacency_graph.assert_symmetry();

  return adjacency_graph;
}
} // namespace

specfem::assembly::mesh<specfem::dimension::type::dim2>::mesh(
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
        &control_nodes_in,
    const specfem::quadrature::quadratures &quadratures,
    const specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>
        &mesh_adjacency_graph) {

  // Get the number of GLL points
  int ngllz = quadratures.gll.get_N();
  int ngllx = quadratures.gll.get_N();

  // Get the number of spectral element from the tags
  nspec = tags.nspec;

  // Get the number of control nodes in each element from the control nodes
  // struct
  ngnod = control_nodes_in.ngnod;

  this->element_grid = specfem::mesh_entity::element<
      specfem::dimension::type::dim2>(ngllz, ngllx);

  auto &mapping =
      static_cast<specfem::assembly::mesh_impl::mesh_to_compute_mapping<
          specfem::dimension::type::dim2> &>(*this);
  auto &quadrature = static_cast<specfem::assembly::mesh_impl::quadrature<
      specfem::dimension::type::dim2> &>(*this);
  auto &shape_functions =
      static_cast<specfem::assembly::mesh_impl::shape_functions<
          specfem::dimension::type::dim2> &>(*this);
  auto &adjacency_graph =
      static_cast<specfem::assembly::mesh_impl::adjacency_graph<
          specfem::dimension::type::dim2> &>(*this);

  auto &control_nodes =
      static_cast<specfem::assembly::mesh_impl::control_nodes<
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

  adjacency_graph =
      build_assembly_adjacency_graph(nspec, mapping, mesh_adjacency_graph);

  if (adjacency_graph.empty()) {
    this->assemble_legacy(); /// This functions needs to be deprecated after we
                             /// update all databases with adjacency graph
  } else {
    // If the adjacency graph is not empty, we use it to assemble the mesh
    this->assemble();
  }

}

void specfem::assembly::mesh<
    specfem::dimension::type::dim2>::assemble_legacy() {

  const int ngnod = this->ngnod;
  const int nspec = this->nspec;

  const auto xi = this->h_xi;
  const auto gamma = this->h_xi;

  const auto ngllz = this->element_grid.ngllz;
  const auto ngllx = this->element_grid.ngllx;

  const auto shape2D = this->h_shape2D;
  const auto coord = this->h_control_node_coord;

  const int scratch_size =
      specfem::kokkos::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  specfem::kokkos::HostView4d<double> global_coordinates(
      "specfem::assembly::mesh::assemble::global_coordinates", nspec, ngllz,
      ngllx, 2);

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        auto shape_functions =
            specfem::shape_function::shape_function(xi(ix), gamma(iz), ngnod);

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
