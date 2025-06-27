#pragma once

#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>

namespace specfem {
namespace io {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim2 {

/**
 * @brief Reads the optional adjacency-map data if it exists. This
 * should only be called after everything else.
 *
 * @param stream - file input stream of the database
 * @param mesh - the mesh object generated from the database up until this point
 * @param mpi - mpi instance.
 */
void read_adjacency_map(
    std::ifstream &stream,
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::MPI::MPI *mpi);
} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace io
} // namespace specfem
