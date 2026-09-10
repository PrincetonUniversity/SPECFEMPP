#pragma once

#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/**
 * @brief Read local and MPI adjacency sections from a thin globe database.
 *
 * The stream must be positioned at the adjacency section, after boundary
 * surfaces have been read. The local graph is stored as one-based Fortran CSR
 * arrays and converted to zero-based SPECFEM++ element indices. MPI interfaces
 * are then read as neighbor-rank/shared-anchor-node lists, converted to
 * zero-based node ids, and matched to element-level MPI adjacency entries.
 *
 * @param stream Input stream positioned at the globe adjacency section
 * @param mesh Globe mesh whose adjacency graph and MPI interfaces are populated
 * @param nnode Number of global anchor nodes used to validate MPI node ids
 * @throws std::runtime_error if CSR ranges, neighbor indices, MPI node ids, or
 *         cross-rank interface matching are invalid
 */
void read_adjacency_graph(std::ifstream &stream,
                          specfem::mesh::globe3d_mesh &mesh, const int nnode);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
