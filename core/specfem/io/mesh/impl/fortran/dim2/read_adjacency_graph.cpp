
#include "specfem/io/mesh/impl/fortran/dim2/read_adjacency_graph.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/mpi.hpp"
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

  auto &g = graph.local_connections();

  int total_local_adjacencies;
  specfem::io::fortran_read_line(stream, &total_local_adjacencies);

  for (int edge_index = 0; edge_index < total_local_adjacencies; edge_index++) {
    int current_element, neighbor_element;
    int connection_int, orientation_int;
    specfem::io::fortran_read_line(stream, &current_element, &neighbor_element,
                                   &connection_int, &orientation_int);

    const auto connection_type =
        static_cast<specfem::element_connections::type>(connection_int);

    if (connection_type ==
            specfem::element_connections::type::strongly_conforming ||
        connection_type == specfem::element_connections::type::nonconforming) {
      const auto edge_orientation =
          static_cast<specfem::mesh_entity::dim2::type>(orientation_int);
      boost::add_edge(current_element - 1, neighbor_element - 1,
                      EdgeProperties{ connection_type, edge_orientation }, g);
    } else {
      throw std::runtime_error("Unknown connection type in adjacency graph.");
    }
  }

  // Check that the graph is symmetric
  graph.assert_symmetry();

  auto &mpi_connections = graph.mpi_connections();

  int total_mpi_adjacencies;
  specfem::io::fortran_read_line(stream, &total_mpi_adjacencies);

  mpi_connections.resize(total_mpi_adjacencies);

  for (int i = 0; i < total_mpi_adjacencies; i++) {
    int ispec_local, ispec_neighbor, neighbor_partition;
    int connection_int, connection_orientation, neighbor_orientation;
    int local_anchor, neighbor_anchor;

    specfem::io::fortran_read_line(
        stream, &ispec_local, &ispec_neighbor, &neighbor_partition,
        &connection_int, &connection_orientation, &neighbor_orientation,
        &local_anchor, &neighbor_anchor);

    ispec_local -= 1;    // convert to 0-based index
    ispec_neighbor -= 1; // convert to 0-based index
    const auto connection_type =
        static_cast<specfem::element_connections::type>(connection_int);
    if (connection_type ==
        specfem::element_connections::type::strongly_conforming) {
      const auto edge_orientation =
          static_cast<specfem::mesh_entity::dim2::type>(connection_orientation);

      mpi_connections[i] = specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim2>::MPIEdgeProperties{
        connection_type,
        edge_orientation,
        static_cast<size_t>(ispec_local),
        static_cast<size_t>(ispec_neighbor),
        neighbor_partition,
        static_cast<specfem::mesh_entity::dim2::type>(neighbor_orientation),
        static_cast<specfem::mesh_entity::dim2::type>(local_anchor),
        static_cast<specfem::mesh_entity::dim2::type>(neighbor_anchor)
      };
    } else {
      throw std::runtime_error(
          "Unknown connection type in MPI adjacency graph.");
    }
  }

  return graph;
}
