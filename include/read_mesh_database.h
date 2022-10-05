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
std::tuple<int, int, int> read_mesh_database_header(std::ifstream &stream,
                                                    const specfem::MPI *mpi);
specfem::HostView2d<type_real> read_coorg_elements(std::ifstream &stream,
                                                   const int npgeo,
                                                   const specfem::MPI *mpi);
specfem::properties read_mesh_properties(std::ifstream &stream,
                                         specfem::MPI *mpi);
std::tuple<int, type_real, bool>
read_mesh_database_attenuation(std::ifstream &stream, const specfem::MPI *mpi);
specfem::materials::material_ind
read_mesh_database_mato(std::ifstream &stream, const int ngnod, const int nspec,
                        const int numat, specfem::MPI *mpi);
specfem::interfaces::interface read_mesh_database_interfaces(
    std::ifstream &stream, specfem::MPI *mpi);
specfem::boundaries::absorbing_boundary
read_mesh_absorbing_boundaries(std::ifstream &stream, int num_abs_boudary_faces,
                               int nspec, specfem::MPI *mpi);
specfem::boundaries::forcing_boundary
read_mesh_database_acoustic_forcing(std::ifstream &stream,
                                    int nelements_acforcing, int nspec,
                                    specfem::MPI *mpi);
specfem::surfaces::acoustic_free_surface
read_mesh_database_free_surface(std::ifstream &stream,
                                int nelem_acoustic_surface, specfem::MPI *mpi);
void read_mesh_database_coupled(std::ifstream &stream,
                                const int num_fluid_solid_edges,
                                const int num_fluid_poro_edges,
                                const int num_solid_poro_edges,
                                const specfem::MPI *mpi);
specfem::elements::tangential_elements
read_mesh_database_tangential(std::ifstream &stream,
                              int nnodes_tangential_curve);
specfem::elements::axial_elements
read_mesh_database_axial(std::ifstream &stream, int nelem_on_the_axis,
                         int nspec, specfem::MPI *mpi);
// std::tuple<specfem::mesh, std::vector<specfem::material *> >
// read_mesh_database(const std::string filename, specfem::parameters &params,
//                    specfem::MPI *mpi);
} // namespace IO

#endif
