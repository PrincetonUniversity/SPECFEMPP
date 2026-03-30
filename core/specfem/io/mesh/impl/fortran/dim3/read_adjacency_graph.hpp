#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3 {

/**
 * @brief Read 3D adjacency graph from Fortran binary database
 *
 * Reads both local and MPI inter-partition adjacency information from a Fortran
 * binary database file. Constructs an adjacency_graph with:
 *
 * **Local adjacencies**: Element connections within a partition encoded as a
 * Boost adjacency list with EdgeProperties (connection type and orientation).
 *
 * **MPI adjacencies**: Cross-partition element connections with anchor point
 * constraints. Each MPI adjacency includes:
 * - Neighbor MPI rank and element indices
 * - Connection type and orientation
 * - Anchor point indices on both local and remote elements
 *
 * **Anchor points** provide deterministic orientation information for shared
 * MPI surfaces. They are element-absolute corner IDs (18-25 in 0-based C++
 * representation, 19-26 in 1-based Fortran) that establish a canonical
 * reference frame, eliminating rotational ambiguity where corner coordinate
 * sets are equivalent but rotationally distinct. This constraint ensures
 * proper continuity of fluxes and stresses across MPI boundaries, removing
 * the possibility of sliding rotations.*
 * Database format (per partition):
 * 1. Count of local adjacencies
 * 2. For each local adjacency: elem1, elem2, connection_type, orientation
 *    (all with 1-based indexing in database, converted to 0-based internally)
 * 3. Count of MPI adjacencies
 * 4. For each MPI adjacency: local_elem, neighbor_elem, connection_type,
 *    orientation, neighbor_rank, neighbor_local_idx, local_idx,
 *    local_anchor_corner_id (1-based Fortran, 19-26),
 *    neighbor_anchor_corner_id (1-based Fortran, 19-26)
 *
 * @param stream Input file stream positioned at adjacency data
 * @param nspec Number of spectral elements in the partition\n *
 * @return adjacency_graph<dim3> Constructed graph with both local and MPI
 *         adjacencies including anchor point constraints\n *
 * @throws Abort called via exit_MPI-style errors if file format is invalid
 *
 * @see MPIEdgeProperties for anchor point field documentation\n * @see
 * EdgeProperties for local connection properties*/
specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
read_adjacency_graph(std::ifstream &stream, const int nspec);

} // namespace specfem::io::mesh::impl::fortran::dim3
