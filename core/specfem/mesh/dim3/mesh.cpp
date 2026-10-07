#include "specfem/mesh.hpp"

void specfem::mesh::mesh_dim3_base::setup_coupled_interfaces() {
  auto &graph = this->adjacency_graph.local_connections();
  auto &materials = this->materials;

  for (const auto v : boost::make_iterator_range(boost::vertices(graph))) {
    for (const auto e :
         boost::make_iterator_range(boost::out_edges(v, graph))) {
      const auto target = boost::target(e, graph);
      auto &edge_props = graph[e];

      const auto [self_medium, self_property, self_attenuation] =
          materials.get_material_type(v);
      const auto [neighbor_medium, neighbor_property, neighbor_attenuation] =
          materials.get_material_type(target);
      if ((self_medium != neighbor_medium) &&
          (edge_props.connection ==
           specfem::element_connections::type::strongly_conforming)) {
        // Change strongly conforming to weakly conforming if media differ
        edge_props.connection =
            specfem::element_connections::type::weakly_conforming;
      }
    }
  }

  this->adjacency_graph.assert_symmetry();

  return;
}
