#pragma once

#include "IO/fortranio/interface.hpp"
#include "mesh/properties/properties.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {

/*
 * @brief Read properties from mesh database
 *
 * @param stream Input stream
 * @param mpi MPI object
 * @return specfem::mesh::properties Property object
 */
specfem::mesh::properties read_properties(std::ifstream &stream,
                                          const specfem::MPI::MPI *mpi);

} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
