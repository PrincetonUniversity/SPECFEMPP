#pragma once

#include "IO/fortranio/interface.hpp"
#include "mesh/mapping/mapping.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim3 {

/*
 * @brief Read paramters from 3D mesh database
 *
 * @param stream Input stream
 * @param mapping Mapping object
 * @param mpi MPI object
 * @return specfem::mesh::mapping<specfem::dimension::type::dim3> Mapping
 * parameters
 */

void read_ibool(std::ifstream &stream,
                specfem::mesh::mapping<specfem::dimension::type::dim3> &mapping,
                const specfem::MPI::MPI *mpi);

} // namespace dim3
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
