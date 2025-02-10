#pragma once

#include "IO/fortranio/interface.hpp"
#include "mesh/parameters/parameters.hpp"
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
 * @param mpi MPI object
 * @return specfem::mesh::parameters<specfem::dimension::type::dim2> Mesh
 * parameters
 */
specfem::mesh::parameters<specfem::dimension::type::dim3>
read_mesh_parameters(std::ifstream &stream, const specfem::MPI::MPI *mpi);

} // namespace dim3
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
