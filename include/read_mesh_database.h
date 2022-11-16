#ifndef READ_MESH_DATABASE_H
#define READ_MESH_DATABASE_H

#include "../include/fortran_IO.h"
#include "../include/material.h"
#include "../include/mesh.h"
#include "../include/params.h"
#include "../include/specfem_mpi.h"
#include <fstream>
#include <iostream>

namespace IO {
void read_mesh_database_header(std::ifstream &stream, specfem::mesh &mesh,
                               specfem::MPI *mpi);
void read_coorg_elements(std::ifstream &stream, specfem::mesh &mesh,
                         specfem::MPI *mpi);
void read_mesh_database_attenuation(std::ifstream &stream,
                                    specfem::parameters &params,
                                    specfem::MPI *mpi);
void read_mesh_database_mato(std::ifstream &stream, specfem::mesh &mesh,
                             specfem::MPI *mpi);
void read_mesh_database_interfaces(std::ifstream &stream,
                                   specfem::interface &interface,
                                   specfem::MPI *mpi);
void read_mesh_absorbing_boundaries(std::ifstream &stream,
                                    specfem::absorbing_boundary &abs_boundary,
                                    int &num_abs_boudary_faces, int nspec,
                                    specfem::MPI *mpi);
void read_mesh_database_acoustic_forcing(
    std::ifstream &stream, specfem::forcing_boundary &acforcing_boundary,
    int nelements_acforcing, int nspec, specfem::MPI *mpi);
void read_mesh_database_free_surface(
    std::ifstream &stream, specfem::acoustic_free_surface &acfree_surface,
    int nelem_acoustic_surface, specfem::MPI *mpi);
void read_mesh_database_coupled(std::ifstream &stream,
                                int num_fluid_solid_edges,
                                int num_fluid_poro_edges,
                                int num_solid_poro_edges, specfem::MPI *mpi);
void read_mesh_database_tangential(
    std::ifstream &stream, specfem::tangential_elements &tangential_nodes,
    int nnodes_tangential_curve);
void read_mesh_database_axial(std::ifstream &stream,
                              specfem::axial_elements &axial_nodes,
                              int nelem_on_the_axis, int nspec,
                              specfem::MPI *mpi);
void read_mesh_database(const std::string filename, specfem::mesh &mesh,
                        specfem::parameters &params,
                        std::vector<specfem::material> &materials,
                        specfem::MPI *mpi);
} // namespace IO

#endif
