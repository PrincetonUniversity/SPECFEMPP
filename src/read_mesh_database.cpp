#include "../include/read_mesh_database.h"
#include "../include/config.h"
#include "../include/fortran_IO.h"
#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/params.h"
#include "../include/read_material_properties.h"
#include "../include/specfem_mpi.h"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <tuple>

std::tuple<int, int, int>
IO::fortran_database::read_mesh_database_header(std::ifstream &stream,
                                                const specfem::MPI::MPI *mpi) {
  // This subroutine reads header values of the database which are skipped
  std::string dummy_s;
  int dummy_i, dummy_i1, dummy_i2;
  type_real dummy_d, dummy_d1;
  bool dummy_b, dummy_b1, dummy_b2, dummy_b3;
  int nspec, npgeo, nproc;

  IO::fortran_IO::fortran_read_line(stream, &dummy_s); // title
  IO::fortran_IO::fortran_read_line(stream, &dummy_i,
                                    &dummy_b); // noise tomography,
                                               // undo_attenuation_and_or_PML
  IO::fortran_IO::fortran_read_line(stream, &nspec); // nspec
  IO::fortran_IO::fortran_read_line(stream, &npgeo,
                                    &nproc); // ngeo, nproc
  IO::fortran_IO::fortran_read_line(stream, &dummy_b1,
                                    &dummy_b2); // output_grid_per_plot,
                                                // interpol
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_i); // ntstep_between_output_info
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_i); // ntstep_between_output_seismos
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_i); // ntstep_between_output_images
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // PML-boundary
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // rotate_PML_activate
  IO::fortran_IO::fortran_read_line(stream, &dummy_d); // rotate_PML_angle
  IO::fortran_IO::fortran_read_line(stream, &dummy_d); // k_min_pml
  IO::fortran_IO::fortran_read_line(stream, &dummy_d); // m_max_PML
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_d); // damping_change_factor_acoustic
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_d); // damping_change_factor_elastic
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_b); // PML_parameter_adjustment
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // read_external_mesh
  IO::fortran_IO::fortran_read_line(stream, &dummy_i); // nelem_PML_thickness
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_i, &dummy_i1,
      &dummy_i2); // NTSTEP_BETWEEN_OUTPUT_SAMPLE,imagetype_JPEG,imagetype_wavefield_dumps
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b1,
      &dummy_b2); // output_postscript_snapshot,output_color_image
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_b2, &dummy_d, &dummy_i1,
      &dummy_d1); // meshvect,modelvect,boundvect,cutsnaps,subsamp_postscript,sizemax_arrows
  IO::fortran_IO::fortran_read_line(stream, &dummy_d); // anglerec
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // initialfield
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_b2,
      &dummy_b3); // add_Bielak_conditions_bottom,add_Bielak_conditions_right,add_Bielak_conditions_top,add_Bielak_conditions_left
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_s,
      &dummy_i); // seismotype,imagetype_postscript
  IO::fortran_IO::fortran_read_line(stream, &dummy_s); // model
  IO::fortran_IO::fortran_read_line(stream, &dummy_s); // save_model
  IO::fortran_IO::fortran_read_line(stream, &dummy_s); // Tomography file
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_i,
      &dummy_b2); // output_grid_ASCII,OUTPUT_ENERGY,NTSTEP_BETWEEN_OUTPUT_ENERGY,output_wavefield_dumps
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_b); // use_binary_for_wavefield_dumps
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b, &dummy_b1,
      &dummy_b2); // ATTENUATION_VISCOELASTIC,ATTENUATION_PORO_FLUID_PART,ATTENUATION_VISCOACOUSTIC
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // use_solvopt
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // save_ASCII_seismograms
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b,
      &dummy_b1); // save_binary_seismograms_single,save_binary_seismograms_double
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_b); // USE_TRICK_FOR_BETTER_PRESSURE
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b); // COMPUTE_INTEGRATED_ENERGY_FIELD
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // save_ASCII_kernels
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_i); // NTSTEP_BETWEEN_COMPUTE_KERNELS
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // APPROXIMATE_HESS_KL
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_b); // NO_BACKWARD_RECONSTRUCTION
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_b); // DRAW_SOURCES_AND_RECEIVERS
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_d,
      &dummy_d1); // Q0_poroelastic,freq0_poroelastic
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // AXISYM
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // psv
  IO::fortran_IO::fortran_read_line(stream, &dummy_d); // factor_subsample_image
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_b); // USE_CONSTANT_MAX_AMPLITUDE
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_d); // CONSTANT_MAX_AMPLITUDE_TO_USE
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b); // USE_SNAPSHOT_NUMBER_IN_FILENAME
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // DRAW_WATER_IN_BLUE
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // US_LETTER
  IO::fortran_IO::fortran_read_line(stream, &dummy_d); // POWER_DISPLAY_COLOR
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // SU_FORMAT
  IO::fortran_IO::fortran_read_line(stream, &dummy_d); // USER_T0
  IO::fortran_IO::fortran_read_line(stream, &dummy_i); // time_stepping_scheme
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_b); // ADD_PERIODIC_CONDITIONS
  IO::fortran_IO::fortran_read_line(stream, &dummy_d); // PERIODIC_HORIZ_DIST
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // GPU_MODE
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_i); // setup_with_binary_database
  IO::fortran_IO::fortran_read_line(stream, &dummy_i, &dummy_d), // NSTEP,DT
      IO::fortran_IO::fortran_read_line(stream,
                                        &dummy_i);     // NT_DUMP_ATTENUATION
  IO::fortran_IO::fortran_read_line(stream, &dummy_b); // ACOUSTIC_FORCING
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_i); // NUMBER_OF_SIMULTANEOUS_RUNS
  IO::fortran_IO::fortran_read_line(stream,
                                    &dummy_b); // BROADCAST_SAME_MESH_AND_MODEL
  IO::fortran_IO::fortran_read_line(
      stream, &dummy_b1,
      &dummy_b2); // ADD_RANDOM_PERTURBATION_TO_THE_MESH,ADD_PERTURBATION_AROUND_SOURCE_ONLY

  mpi->sync_all();

  return std::make_tuple(nspec, npgeo, nproc);
}

specfem::kokkos::HostView2d<type_real>
IO::fortran_database::read_coorg_elements(std::ifstream &stream,
                                          const int npgeo,
                                          const specfem::MPI::MPI *mpi) {

  int ipoin = 0;

  type_real coorgi, coorgj;
  int buffer_length;
  specfem::kokkos::HostView2d<type_real> coorg("specfem::mesh::coorg", ndim,
                                               npgeo);

  for (int i = 0; i < npgeo; i++) {
    IO::fortran_IO::fortran_read_line(stream, &ipoin, &coorgi, &coorgj);
    if (ipoin < 1 || ipoin > npgeo) {
      throw std::runtime_error("Error reading coordinates");
    }
    // coorg stores the x,z for every control point
    // coorg([0, 2), i) = [x, z]
    coorg(0, ipoin - 1) = coorgi;
    coorg(1, ipoin - 1) = coorgj;
  }

  return coorg;
}

std::tuple<int, type_real, bool>
IO::fortran_database::read_mesh_database_attenuation(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  int n_sls;
  type_real attenuation_f0_reference;
  bool read_velocities_at_f0;
  IO::fortran_IO::fortran_read_line(stream, &n_sls, &attenuation_f0_reference,
                                    &read_velocities_at_f0);

  if (n_sls < 1) {
    throw std::runtime_error("must have N_SLS >= 1 even if attenuation if off "
                             "because it is used to assign some arrays");
  }

  return std::make_tuple(n_sls, attenuation_f0_reference,
                         read_velocities_at_f0);
}

void IO::fortran_database::read_mesh_database_coupled(
    std::ifstream &stream, const int num_fluid_solid_edges,
    const int num_fluid_poro_edges, const int num_solid_poro_edges,
    const specfem::MPI::MPI *mpi) {

  int dummy_i, dummy_i1;

  if (num_fluid_solid_edges > 0 || num_fluid_poro_edges > 0 ||
      num_solid_poro_edges > 0) {
    mpi->cout("\n Warning coupled surfaces haven't been implemented yet \n");
  }

  if (num_fluid_solid_edges > 0) {
    for (int inum = 0; inum < num_fluid_solid_edges; inum++)
      IO::fortran_IO::fortran_read_line(stream, &dummy_i, &dummy_i1);
  }

  if (num_fluid_poro_edges > 0) {
    for (int inum = 0; inum < num_fluid_poro_edges; inum++)
      IO::fortran_IO::fortran_read_line(stream, &dummy_i, &dummy_i1);
  }

  if (num_solid_poro_edges > 0) {
    for (int inum = 0; inum < num_solid_poro_edges; inum++)
      IO::fortran_IO::fortran_read_line(stream, &dummy_i, &dummy_i1);
  }
  return;
}
