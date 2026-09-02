#pragma once

#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/** @brief Read local and MPI adjacency data from a globe database. */
void read_adjacency_graph(std::ifstream &stream,
                          specfem::mesh::globe3d_mesh &mesh, const int nnode);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
