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
   * Phase 2: MPI adjacencies with anchor points (7-field format)
   * Reads inter-partition adjacencies enriched with anchor point constraints.
   * For each MPI connection, reads 7 fields:
   * - Local element index
   * - Neighbor MPI rank
   * - Neighbor element index
   * - Local connection ID (interface/edge type)
   * - Neighbor connection ID (interface/edge type)
   * - Local anchor corner ID (element-absolute, Fortran 1-based 19-26)
   * - Neighbor anchor corner ID (element-absolute, Fortran 1-based 19-26)
   *
   * Anchor points are element-absolute corner IDs (typed as
   * specfem::mesh_entity::dim3::type, range 19-26) that match between local
   * and remote elements via coordinate equivalence. They establish a canonical
   * orientation by identifying specific corners on the hexahedron geometry,
   * removing the possibility of rotational sliding across MPI boundaries by
   * creating an additional constraint beyond corner coordinate matching.
   *
   * Corner ID mapping (element-absolute, Fortran 1-based, matching
   * specfem::mesh_entity::dim3::type convention):
   *   19: bottom_front_left      20: bottom_front_right
   *   21: bottom_back_left       22: bottom_back_right
   *   23: top_front_left         24: top_front_right
   *   25: top_back_left          26: top_back_right
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

  using MPIEdgeProperties = specfem::mesh::adjacency_graph<
      specfem::element::dimension_tag::dim3>::MPIEdgeProperties;

  for (int i = 0; i < mpi_adjacencies; ++i) {
    int local_elem, neighbor_rank, neighbor_elem, local_conn_id,
        neighbor_conn_id, local_anchor, neighbor_anchor;

    specfem::io::fortran_read_line(
        stream, &local_elem, &neighbor_rank, &neighbor_elem, &local_conn_id,
        &neighbor_conn_id, &local_anchor, &neighbor_anchor);

    // Convert to zero-based indexing
    local_elem -= 1;
    neighbor_elem -= 1;

    MPIEdgeProperties mpi_edge_props(
        specfem::element_connections::type::strongly_conforming,
        static_cast<specfem::mesh_entity::dim3::type>(local_conn_id),
        neighbor_rank,
        static_cast<specfem::mesh_entity::dim3::type>(neighbor_conn_id),
        local_elem, neighbor_elem,
        static_cast<specfem::mesh_entity::dim3::type>(local_anchor),
        static_cast<specfem::mesh_entity::dim3::type>(neighbor_anchor));
    mpi_conns.push_back(mpi_edge_props);
  }

  return graph;
}
