#include "specfem/mesh.hpp"

void specfem::mesh::mesh<
    specfem::dimension::type::dim3>::setup_coupled_interfaces() {
  auto &graph = this->adjacency_graph.graph();
  auto &materials = this->materials;

  for (const auto v : boost::make_iterator_range(boost::vertices(graph))) {
    for (const auto e :
         boost::make_iterator_range(boost::out_edges(v, graph))) {
      const auto target = boost::target(e, graph);
      auto &edge_props = graph[e];

      const auto [self_medium, self_property] = materials.get_material_type(v);
      const auto [neighbor_medium, neighbor_property] =
          materials.get_material_type(target);
      if ((self_medium != neighbor_medium) &&
          (edge_props.connection ==
           specfem::connections::type::strongly_conforming)) {
        // Change strongly conforming to weakly conforming if media differ
        edge_props.connection = specfem::connections::type::weakly_conforming;
      }
    }
  }

  this->adjacency_graph.assert_symmetry();

  return;
}
