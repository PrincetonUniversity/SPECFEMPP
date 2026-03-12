#pragma once

#include "adjacency_graph.hpp"
#include "mesh_to_compute_mapping.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/iteration_macros.hpp>

specfem::assembly::mesh_impl::
    adjacency_graph<specfem::element::dimension_tag::dim3>::adjacency_graph(
        const int nspec,
        const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
            dimension_tag> &mapping,
        const specfem::mesh::adjacency_graph<dimension_tag>
            &mesh_adjacency_graph) {

  auto &g = this->graph();
  const auto &mesh_g = mesh_adjacency_graph.graph();

  for (int ispec = 0; ispec < nspec; ispec++) {
    // Get mesh index
    const int ispec_mesh = mapping.h_compute_to_mesh(ispec);
    // Iterate over all outgoing edges
    for (auto iedge :
         boost::make_iterator_range(boost::out_edges(ispec_mesh, mesh_g))) {
      // Get the target mesh index
      const int target_ispec_mesh = boost::target(iedge, mesh_g);
      // Get the target specfem index
      const int target_ispec = mapping.h_mesh_to_compute(target_ispec_mesh);
      // Get edge property
      const auto edge_property = mesh_g[iedge];
      // Add the edge to the adjacency graph
      boost::add_edge(ispec, target_ispec, edge_property, g);
    }
  }

  // Check that the graph is symmetric
  this->assert_symmetry();
  return;
}
