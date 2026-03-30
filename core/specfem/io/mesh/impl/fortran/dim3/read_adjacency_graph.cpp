#include "specfem/io/mesh/impl/fortran/dim3/read_adjacency_graph.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/mesh.hpp"

specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
specfem::io::mesh::impl::fortran::dim3::read_adjacency_graph(
    std::ifstream &stream, const int nspec) {

  specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3> graph(
      nspec);

  auto &g = graph.local_connections();

  int total_adjacencies;
  specfem::io::fortran_read_line(stream, &total_adjacencies);

  using EdgeProperties = specfem::mesh::adjacency_graph<
      specfem::element::dimension_tag::dim3>::EdgeProperties;

  for (int i = 0; i < total_adjacencies; ++i) {
    int elem1, elem2;
    int connection_type_int, orientation_int;

    specfem::io::fortran_read_line(stream, &elem1, &elem2, &connection_type_int,
                                   &orientation_int);

    // Convert to zero-based indexing
    elem1 -= 1;
    elem2 -= 1;

    EdgeProperties edge_props(
        static_cast<specfem::element_connections::type>(connection_type_int),
        static_cast<specfem::mesh_entity::dim3::type>(orientation_int));
    boost::add_edge(elem1, elem2, edge_props, g);
  }

  graph.assert_symmetry();

  int mpi_adjacencies;

  auto &mpi_conns = graph.mpi_connections();

  specfem::io::fortran_read_line(stream, &mpi_adjacencies);

  for (int i = 0; i < mpi_adjacencies; ++i) {
    int elem1, elem2;
    int connection_type_int, orientation_int;
    int neighbor_rank, neighbor_local_idx, local_idx, local_anchor_idx,
        neighbor_anchor_idx;

    specfem::io::fortran_read_line(stream, &elem1, &elem2, &connection_type_int,
                                   &orientation_int, &neighbor_rank,
                                   &neighbor_local_idx, &local_idx,
                                   &local_anchor_idx, &neighbor_anchor_idx);

    // Convert to zero-based indexing
    elem1 -= 1;
    elem2 -= 1;

    MPIEdgeProperties mpi_edge_props(
        static_cast<specfem::element_connections::type>(connection_type_int),
        static_cast<specfem::mesh_entity::dim3::type>(orientation_int),
        neighbor_rank, neighbor_local_idx, local_idx, local_anchor_idx,
        neighbor_anchor_idx);
    mpi_conns.push_back(mpi_edge_props);
  }

  return graph;
}
