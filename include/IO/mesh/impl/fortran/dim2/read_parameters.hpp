#pragma once

#include "IO/fortranio/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim2 {
/*
 * @brief Read paramters from 2D mesh database
 *
 * @param stream Input stream
 * @param mpi MPI object
 * @return specfem::mesh::parameters<specfem::dimension::type::dim2> Mesh
 * parameters
 */
specfem::mesh::parameters<specfem::dimension::type::dim2>
read_mesh_parameters(std::ifstream &stream, const specfem::MPI::MPI *mpi);

} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
