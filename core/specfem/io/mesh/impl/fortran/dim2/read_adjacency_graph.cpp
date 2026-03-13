
#include "specfem/io/mesh/impl/fortran/dim2/read_adjacency_graph.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <map>
#include <sstream>

specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim2>
specfem::io::mesh::impl::fortran::dim2::read_adjacency_graph(
    const int nspec, std::ifstream &stream) {

  using EdgeProperties = specfem::mesh::adjacency_graph<
      specfem::element::dimension_tag::dim2>::EdgeProperties;

  specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim2> graph(
      nspec);

  auto &g = graph.graph();

  int total_adjacencies;
  specfem::io::fortran_read_line(stream, &total_adjacencies);

  for (int edge_index = 0; edge_index < total_adjacencies; edge_index++) {
    int current_element, neighbor_element;
    int connection_int, orientation_int;
    int neighbor_partition;
    specfem::io::fortran_read_line(stream, &current_element, &neighbor_element,
                                   &connection_int, &orientation_int,
                                   &neighbor_partition);

    const auto connection_type =
        static_cast<specfem::element_connections::type>(connection_int);

    if (connection_type ==
            specfem::element_connections::type::strongly_conforming ||
        connection_type == specfem::element_connections::type::nonconforming) {
      const auto edge_orientation =
          static_cast<specfem::mesh_entity::dim2::type>(orientation_int);
      boost::add_edge(current_element - 1, neighbor_element - 1,
                      EdgeProperties{ connection_type, edge_orientation,
                                      neighbor_partition },
                      g);
    } else {
      throw std::runtime_error("Unknown connection type in adjacency graph.");
    }
  }

  // Check that the graph is symmetric
  graph.assert_symmetry();

  return graph;
}
