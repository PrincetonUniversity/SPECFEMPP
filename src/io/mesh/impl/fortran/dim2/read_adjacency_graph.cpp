
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

    if (connection_type == specfem::connections::type::strongly_conforming) {
      const auto edge_orientation =
          static_cast<specfem::mesh_entity::type>(orientation_int);
      boost::add_edge(
          current_element - 1, neighbor_element - 1,
          EdgeProperties{ specfem::connections::type::strongly_conforming,
                          edge_orientation },
          g);
    } else {
      throw std::runtime_error("Unknown connection type in adjacency graph.");
    }
  }

  // Check that the graph is symmetric
  for (const auto &edge : boost::make_iterator_range(boost::edges(g))) {
    const auto source = boost::source(edge, g);
    const auto target = boost::target(edge, g);
    if (!boost::edge(target, source, g).second) {
      std::ostringstream message;
      message << "Adjacency graph is not symmetric: edge from " << source
              << " to " << target << " exists, but not from " << target
              << " to " << source;
      throw std::runtime_error(message.str());
    }
  }

  return graph;
}
