#pragma once

#include "mesh/properties/properties.hpp"
#include "IO/fortranio/interface.hpp"

namespace specfem {
namespace IO {
namespace mesh {
namespace fortran {

/*
* @brief Read properties from mesh database
*
* @param stream Input stream
* @param mpi MPI object
* @return specfem::mesh::properties
*/
specfem::mesh::properties read_properties(std::ifstream &stream,
                                      const specfem::MPI::MPI *mpi) {};

} // namespace fortran
} // namespace mesh
} // namespace IO
} // namespace specfem