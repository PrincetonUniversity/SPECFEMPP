#pragma once

#include "IO/fortranio/interface.hpp"
#include "mesh/coordinates/coordinates.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim3 {

/*
 * @brief Read coordinates from 3D mesh database
 *
 * @param stream Input stream
 * @param coordinates Coordinates object
 * @param mpi MPI object
 */
void read_xyz(
    std::ifstream &stream,
    specfem::mesh::coordinates<specfem::dimension::type::dim3> &coordinates,
    const specfem::MPI::MPI *mpi);

void read_jacobian(
    std::ifstream &stream,
    specfem::mesh::coordinates<specfem::dimension::type::dim3> &coordinates,
    const specfem::MPI::MPI *mpi);

} // namespace dim3
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
