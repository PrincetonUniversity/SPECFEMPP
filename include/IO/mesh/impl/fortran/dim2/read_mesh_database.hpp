#pragma once

#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <fstream>
#include <iostream>
#include <tuple>

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim2 {

/**
 * @brief Read fortran bindary database header.
 *
 * This section populates nspec, npgeo, nproc in the mesh struct
 *
 * @param stream Stream object for fortran binary file buffered to header
 * section
 * @param mpi Pointer to MPI object
 * @return std::tuple<int, int, int> nspec, npgeo, nproc values read from
 * database file
 */
std::tuple<int, int, int>
read_mesh_database_header(std::ifstream &stream, const specfem::MPI::MPI *mpi);
/**
 * @brief Read coorg elements from fortran binary database file
 *
 * @param stream Stream object for fortran binary file buffered to header
 * section
 * @param npgeo Total number of control nodes in simulation box
 * @param mpi Pointer to MPI object
 * @return std::tuple<int, int, int>  nspec, npgeo, nproc values read from
 */
specfem::kokkos::HostView2d<type_real>
read_coorg_elements(std::ifstream &stream, const int npgeo,
                    const specfem::MPI::MPI *mpi);

/**
 * @warning These two routines need to be implemented
 */

std::tuple<int, type_real, bool>
read_mesh_database_attenuation(std::ifstream &stream,
                               const specfem::MPI::MPI *mpi);

} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
