#pragma once

#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>

namespace specfem {
namespace io {
namespace mesh {
namespace impl {

constexpr int FOOTERCODE_END = 0;
constexpr int FOOTERCODE_ADJACENCYMAP = 1;

namespace fortran {
namespace dim2 {

/**
 * @brief Reads any auxiliary information stored inside the mesh database. This
 * should only be called after everything else.
 *
 * @param stream - file input stream of the database
 * @param mesh - the mesh object generated from the database up until this point
 * @param mpi - mpi instance.
 */
void read_footer(std::ifstream &stream,
                 specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
                 const specfem::MPI::MPI *mpi);
} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace io
} // namespace specfem
