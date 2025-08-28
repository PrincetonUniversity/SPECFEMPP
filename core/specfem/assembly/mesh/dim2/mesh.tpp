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
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

namespace {
struct qp {
  type_real x = 0, z = 0;
  int iloc = 0, iglob = 0;
};

type_real get_tolerance(std::vector<qp> cart_cord, const int nspec,
                        const int ngllxz) {

  assert(cart_cord.size() == ngllxz * nspec);

  type_real xtypdist = std::numeric_limits<type_real>::max();
  for (int ispec = 0; ispec < nspec; ispec++) {
    type_real xmax = std::numeric_limits<type_real>::min();
    type_real xmin = std::numeric_limits<type_real>::max();
    type_real ymax = std::numeric_limits<type_real>::min();
    type_real ymin = std::numeric_limits<type_real>::max();
    for (int xz = 0; xz < ngllxz; xz++) {
      int iloc = ispec * (ngllxz) + xz;
      xmax = std::max(xmax, cart_cord[iloc].x);
      xmin = std::min(xmin, cart_cord[iloc].x);
      ymax = std::max(ymax, cart_cord[iloc].z);
      ymin = std::min(ymin, cart_cord[iloc].z);
    }

    xtypdist = std::min(xtypdist, xmax - xmin);
    xtypdist = std::min(xtypdist, ymax - ymin);
  }

  return 1e-6 * xtypdist;
}

specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2>
assign_numbering(specfem::kokkos::HostView4d<double> global_coordinates) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxz = ngll * ngll;

  std::vector<qp> cart_cord(nspec * ngllxz);

  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;

  int iloc = 0;
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          cart_cord[iloc].x = global_coordinates(ispec, iz, ix, 0);
          cart_cord[iloc].z = global_coordinates(ispec, iz, ix, 1);
          cart_cord[iloc].iloc = iloc;
          iloc++;
        }
      }
    }
  }

  // Sort cartesian coordinates in ascending order i.e.
  // cart_cord = [{0,0}, {0, 25}, {0, 50}, ..., {50, 0}, {50, 25}, {50, 50}]
  std::sort(cart_cord.begin(), cart_cord.end(),
            [&](const qp qp1, const qp qp2) {
              if (qp1.x != qp2.x) {
                return qp1.x < qp2.x;
              }

              return qp1.z < qp2.z;
            });

  // Setup numbering
  int ig = 0;
  cart_cord[0].iglob = ig;

  type_real xtol = get_tolerance(cart_cord, nspec, ngllxz);

  for (int iloc = 1; iloc < cart_cord.size(); iloc++) {
    // check if the previous point is same as current
    if ((std::abs(cart_cord[iloc].x - cart_cord[iloc - 1].x) > xtol) ||
        (std::abs(cart_cord[iloc].z - cart_cord[iloc - 1].z) > xtol)) {
      ig++;
    }
    cart_cord[iloc].iglob = ig;
  }

  std::vector<qp> copy_cart_cord(nspec * ngllxz);

  // reorder cart cord in original format
  for (int i = 0; i < cart_cord.size(); i++) {
    int iloc = cart_cord[i].iloc;
    copy_cart_cord[iloc] = cart_cord[i];
  }

  int nglob = ig + 1;

  specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2> points(
      nspec, ngll, ngll, nglob);

  // Assign numbering to corresponding ispec, iz, ix
  std::vector<int> iglob_counted(nglob, -1);
  iloc = 0;
  int inum = 0;
  type_real xmin = std::numeric_limits<type_real>::max();
  type_real xmax = std::numeric_limits<type_real>::min();
  type_real zmin = std::numeric_limits<type_real>::max();
  type_real zmax = std::numeric_limits<type_real>::min();

  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          if (iglob_counted[copy_cart_cord[iloc].iglob] == -1) {

            const type_real x_cor = copy_cart_cord[iloc].x;
            const type_real z_cor = copy_cart_cord[iloc].z;
            if (xmin > x_cor)
              xmin = x_cor;
            if (zmin > z_cor)
              zmin = z_cor;
            if (xmax < x_cor)
              xmax = x_cor;
            if (zmax < z_cor)
              zmax = z_cor;

            iglob_counted[copy_cart_cord[iloc].iglob] = inum;
            points.h_index_mapping(ispec, iz, ix) = inum;
            points.h_coord(0, ispec, iz, ix) = x_cor;
            points.h_coord(1, ispec, iz, ix) = z_cor;
            inum++;
          } else {
            points.h_index_mapping(ispec, iz, ix) =
                iglob_counted[copy_cart_cord[iloc].iglob];
            points.h_coord(0, ispec, iz, ix) = copy_cart_cord[iloc].x;
            points.h_coord(1, ispec, iz, ix) = copy_cart_cord[iloc].z;
          }
          iloc++;
        }
      }
    }
  }

  points.xmin = xmin;
  points.xmax = xmax;
  points.zmin = zmin;
  points.zmax = zmax;

  assert(nglob != (nspec * ngllxz));

  assert(inum == nglob);

  Kokkos::deep_copy(points.index_mapping, points.h_index_mapping);
  Kokkos::deep_copy(points.coord, points.h_coord);

  return points;
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
  nspec = tags.nspec;
  ngllz = quadratures.gll.get_N();
  ngllx = quadratures.gll.get_N();
  ngnod = control_nodes_in.ngnod;

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
