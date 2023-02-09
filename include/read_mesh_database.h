#ifndef READ_MESH_DATABASE_H
#define READ_MESH_DATABASE_H

#include "../include/boundaries.h"
#include "../include/elements.h"
#include "../include/fortran_IO.h"
#include "../include/material.h"
#include "../include/material_indic.h"
#include "../include/mesh_properties.h"
#include "../include/mpi_interfaces.h"
#include "../include/params.h"
#include "../include/specfem_mpi.h"
#include "../include/surfaces.h"
#include <fstream>
#include <iostream>
#include <tuple>

namespace IO {
/**
 * Helper routines to read fortran binary database
 *
 */
namespace fortran_database {

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
 * @return specfem::kokkos::HostView2d<type_real> coorg values as read from
 * fortran binary database file
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

void read_mesh_database_coupled(std::ifstream &stream,
                                const int num_fluid_solid_edges,
                                const int num_fluid_poro_edges,
                                const int num_solid_poro_edges,
                                const specfem::MPI::MPI *mpi);
} // namespace fortran_database
} // namespace IO

#endif
