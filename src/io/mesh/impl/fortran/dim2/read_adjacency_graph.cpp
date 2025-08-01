
#include "io/mesh/impl/fortran/dim2/read_adjacency_graph.hpp"
#include "enumerations/interface.hpp"
#include "io/fortranio/interface.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <map>

enum class ConnectionType {
  STRONGLY_CONNECTED_EDGE = 1,
  STRONGLY_CONNECTED_VERTEX = 2
};

enum class EdgeOrientation { BOTTOM = 1, RIGHT = 2, TOP = 3, LEFT = 4 };

enum class VertexOrientation {
  BOTTOM_LEFT = 1,
  BOTTOM_RIGHT = 2,
  TOP_RIGHT = 3,
  TOP_LEFT = 4
};

const static std::map<EdgeOrientation, specfem::connections::orientation>
    connection_type_map = {
      { EdgeOrientation::BOTTOM, specfem::connections::orientation::bottom },
      { EdgeOrientation::RIGHT, specfem::connections::orientation::right },
      { EdgeOrientation::TOP, specfem::connections::orientation::top },
      { EdgeOrientation::LEFT, specfem::connections::orientation::left }
    };

const static std::map<VertexOrientation, specfem::connections::orientation>
    vertex_orientation_map = {
      { VertexOrientation::BOTTOM_LEFT,
        specfem::connections::orientation::bottom_left },
      { VertexOrientation::BOTTOM_RIGHT,
        specfem::connections::orientation::bottom_right },
      { VertexOrientation::TOP_RIGHT,
        specfem::connections::orientation::top_right },
      { VertexOrientation::TOP_LEFT,
        specfem::connections::orientation::top_left }
    };

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

    const auto connection_type = static_cast<ConnectionType>(connection_int);

    if (connection_type == ConnectionType::STRONGLY_CONNECTED_EDGE) {
      const auto edge_orientation =
          static_cast<EdgeOrientation>(orientation_int);
      const auto specfem_orientation = connection_type_map.at(edge_orientation);
      boost::add_edge(
          current_element - 1, neighbor_element - 1,
          EdgeProperties{ specfem::connections::type::strongly_conforming,
                          specfem_orientation },
          g);
    } else if (connection_type == ConnectionType::STRONGLY_CONNECTED_VERTEX) {
      const auto vertex_orientation =
          static_cast<VertexOrientation>(orientation_int);
      const auto specfem_orientation =
          vertex_orientation_map.at(vertex_orientation);
      boost::add_edge(
          current_element - 1, neighbor_element - 1,
          EdgeProperties{ specfem::connections::type::strongly_conforming,
                          specfem_orientation },
          g);
    } else {
      throw std::runtime_error("Unknown connection type in adjacency graph.");
    }
  }

  return graph;
}
