#include "specfem/io/mesh/impl/fortran/dim3/read_adjacency_graph.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/mesh.hpp"

specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
specfem::io::mesh::impl::fortran::dim3::read_adjacency_graph(
    std::ifstream &stream, const int nspec) {

  /**
   * Two-phase read of adjacency data from Fortran database:
   *
   * Phase 1: Local adjacencies
   * Reads intra-partition element adjacencies and populates the local Boost
   * graph. Connection type (conforming/non-conforming) and orientation
   * (face/edge type) are stored in EdgeProperties. Database uses 1-based
   * indexing; converted to 0-based for internal representation.
   *
   * Phase 2: MPI adjacencies with anchor points
   * Reads inter-partition adjacencies enriched with anchor point constraints.
   * For each MPI connection, reads:
   * - Local and remote element indices
   * - Connection type and orientation
   * - Neighbor MPI rank and remote element's local index
   * - Local anchor corner ID (element-absolute, 1-based Fortran 19-26,
   *   converted to 0-based C++ 18-25)
   * - Remote anchor corner ID (element-absolute, 1-based Fortran 19-26,
   *   converted to 0-based C++ 18-25)
   *
   * Anchor points are element-absolute corner IDs that match between local
   * and remote elements via coordinate equivalence. They establish a canonical
   * orientation by identifying specific corners on the hexahedron geometry,
   * removing the possibility of rotational sliding across MPI boundaries by
   * creating an additional constraint beyond corner coordinate matching.
   *
   * Corner ID mapping (element-absolute, 0-based C++ notation):
   *   18: bottom_front_left      19: bottom_front_right
   *   20: bottom_back_left       21: bottom_back_right
   *   22: top_front_left         23: top_front_right
   *   24: top_back_left          25: top_back_right
   */

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
