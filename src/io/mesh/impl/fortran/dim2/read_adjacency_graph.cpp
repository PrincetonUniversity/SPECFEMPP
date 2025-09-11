
#include "io/mesh/impl/fortran/dim2/read_adjacency_graph.hpp"
#include "enumerations/interface.hpp"
#include "io/fortranio/interface.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <map>
#include <sstream>

specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>
specfem::io::mesh::impl::fortran::dim2::read_adjacency_graph(
    const int nspec, std::ifstream &stream) {

  using EdgeProperties = specfem::mesh::adjacency_graph<
      specfem::dimension::type::dim2>::EdgeProperties;

  bool read_adjacency_graph;
  auto current_position = stream.tellg();
  try {
    specfem::io::fortran_read_line(stream, &read_adjacency_graph);
  } catch (std::runtime_error &e) {
    stream.clear();
    stream.seekg(current_position);
    read_adjacency_graph = false;
  }
  if (!read_adjacency_graph) {
    specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>
        empty_graph{};
    return empty_graph;
  }

  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(nspec);

  auto &g = graph.graph();

  int total_adjacencies;
  specfem::io::fortran_read_line(stream, &total_adjacencies);

  for (int edge_index = 0; edge_index < total_adjacencies; edge_index++) {
    int current_element, neighbor_element;
    int connection_int, orientation_int;
    specfem::io::fortran_read_line(stream, &current_element, &neighbor_element,
                                   &connection_int, &orientation_int);

    const auto connection_type =
        static_cast<specfem::connections::type>(connection_int);

    if (connection_type == specfem::connections::type::strongly_conforming ||
        connection_type == specfem::connections::type::nonconforming) {
      const auto edge_orientation =
          static_cast<specfem::mesh_entity::type>(orientation_int);
      boost::add_edge(current_element - 1, neighbor_element - 1,
                      EdgeProperties{ connection_type, edge_orientation }, g);
    } else {
      throw std::runtime_error("Unknown connection type in adjacency graph.");
    }
  }

  // Check that the graph is symmetric
  graph.assert_symmetry();

  return graph;
}
