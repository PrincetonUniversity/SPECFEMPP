#pragma once

#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"

namespace specfem {

namespace IO {

/* @brief Construct a mesh object from a Fortran binary database file
 *
 * @param filename Fortran binary database filename
 * @param mpi pointer to MPI object to manage communication
 */
specfem::mesh::mesh read_mesh(const std::string filename,
                              const specfem::MPI::MPI *mpi);

} // namespace IO
} // namespace specfem
