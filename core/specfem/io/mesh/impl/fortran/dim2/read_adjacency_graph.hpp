#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"

namespace specfem::io::mesh::impl::fortran::dim2 {

/**
 * @brief Read the 2D adjacency graph from a Fortran binary database stream
 *
 * Reads per-partition adjacency records written by the Fortran mesher.
 * Each record contains 5 integers:
 *   (local_elem1, local_elem2, connection_type, orientation,
 * neighbor_partition)
 *
 * - local_elem1 / local_elem2: 1-based local element indices
 * - connection_type: strongly_conforming or nonconforming
 * - orientation: shared mesh entity orientation (edge / corner identifier)
 * - neighbor_partition: MPI rank owning local_elem2 (-1 for legacy
 *   single-partition databases, otherwise the partition rank)
 *
 * @param nspec Number of spectral elements in this partition
 * @param stream Open Fortran unformatted binary input stream
 * @return Populated adjacency_graph with EdgeProperties including
 *         neighbor_partition
 */
specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim2>
read_adjacency_graph(const int nspec, std::ifstream &stream);

} // namespace specfem::io::mesh::impl::fortran::dim2
