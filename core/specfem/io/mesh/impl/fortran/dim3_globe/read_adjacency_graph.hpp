#pragma once

#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::dim3_globe {

/**
 * @brief Read local and MPI adjacency sections from a thin globe database.
 *
 * The stream must be positioned at the adjacency section, after boundary
 * surfaces have been read. The local graph is stored as one-based Fortran CSR
 * arrays and converted to zero-based SPECFEM++ element indices. Resolved MPI
 * element adjacencies use the same seven-field representation as Cartesian 3-D
 * databases.
 *
 * @param stream Input stream positioned at the globe adjacency section
 * @param mesh Globe mesh whose local and MPI adjacency entries are populated
 * @throws std::runtime_error if local or MPI adjacency records are invalid
 */
void read_adjacency_graph(std::ifstream &stream,
                          specfem::mesh::globe3d_mesh &mesh);

} // namespace specfem::io::dim3_globe
