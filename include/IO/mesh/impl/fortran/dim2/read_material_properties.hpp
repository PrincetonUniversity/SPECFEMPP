#pragma once

#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim2 {

/**
 * Read material properties from a fotran binary database
 *
 * @param stream Stream object for fortran binary file buffered to materials
 * section
 * @param numat Number of materials to be read
 * @param mpi Pointer to MPI object
 * @return std::vector<specfem::medium *> Pointer to material objects read
 * from the database file
 */

specfem::mesh::materials read_material_properties(
    std::ifstream &stream, const int numat, const int nspec,
    const specfem::kokkos::HostView2d<int> knods, const specfem::MPI::MPI *mpi);

} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
