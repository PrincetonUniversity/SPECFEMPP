#include "mesh/IO/fortran/read_mesh_database.hpp"
#include "fortran_IO.h"
#include "kokkos_abstractions.h"
#include "mesh/IO/fortran/read_material_properties.hpp"
#include "params.h"
#include "specfem_mpi.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <tuple>

std::tuple<int, int, int> specfem::mesh::IO::fortran::read_mesh_database_header(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {
  // This subroutine reads header values of the database which are skipped
  std::string dummy_s;
  int dummy_i, dummy_i1, dummy_i2;
  type_real dummy_d, dummy_d1;
  bool dummy_b, dummy_b1, dummy_b2, dummy_b3;
  int nspec, npgeo, nproc;

  specfem::fortran_IO::fortran_read_line(stream, &dummy_s); // title
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_i,
      &dummy_b); // noise tomography,
                 // undo_attenuation_and_or_PML
  specfem::fortran_IO::fortran_read_line(stream, &nspec); // nspec
  specfem::fortran_IO::fortran_read_line(stream, &npgeo,
                                         &nproc); // ngeo, nproc
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b1,
                                         &dummy_b2); // output_grid_per_plot,
                                                     // interpol
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_i); // ntstep_between_output_info
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_i); // ntstep_between_output_seismos
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_i); // ntstep_between_output_images
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // PML-boundary
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_b); // rotate_PML_activate
  specfem::fortran_IO::fortran_read_line(stream, &dummy_d); // rotate_PML_angle
  specfem::fortran_IO::fortran_read_line(stream, &dummy_d); // k_min_pml
  specfem::fortran_IO::fortran_read_line(stream, &dummy_d); // m_max_PML
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_d); // damping_change_factor_acoustic
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_d); // damping_change_factor_elastic
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_b); // PML_parameter_adjustment
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_b); // read_external_mesh
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_i); // nelem_PML_thickness
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_i, &dummy_i1,
      &dummy_i2); // NTSTEP_BETWEEN_OUTPUT_SAMPLE,imagetype_JPEG,imagetype_wavefield_dumps
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b1,
      &dummy_b2); // output_postscript_snapshot,output_color_image
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_b2, &dummy_d, &dummy_i1,
      &dummy_d1); // meshvect,modelvect,boundvect,cutsnaps,subsamp_postscript,sizemax_arrows
  specfem::fortran_IO::fortran_read_line(stream, &dummy_d); // anglerec
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // initialfield
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_b2,
      &dummy_b3); // add_Bielak_conditions_bottom,add_Bielak_conditions_right,add_Bielak_conditions_top,add_Bielak_conditions_left
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_s,
      &dummy_i); // seismotype,imagetype_postscript
  specfem::fortran_IO::fortran_read_line(stream, &dummy_s); // model
  specfem::fortran_IO::fortran_read_line(stream, &dummy_s); // save_model
  specfem::fortran_IO::fortran_read_line(stream, &dummy_s); // Tomography file
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_i,
      &dummy_b2); // output_grid_ASCII,OUTPUT_ENERGY,NTSTEP_BETWEEN_OUTPUT_ENERGY,output_wavefield_dumps
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_b); // use_binary_for_wavefield_dumps
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b, &dummy_b1,
      &dummy_b2); // ATTENUATION_VISCOELASTIC,ATTENUATION_PORO_FLUID_PART,ATTENUATION_VISCOACOUSTIC
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // use_solvopt
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_b); // save_ASCII_seismograms
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b,
      &dummy_b1); // save_binary_seismograms_single,save_binary_seismograms_double
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_b); // USE_TRICK_FOR_BETTER_PRESSURE
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b); // COMPUTE_INTEGRATED_ENERGY_FIELD
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_b); // save_ASCII_kernels
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_i); // NTSTEP_BETWEEN_COMPUTE_KERNELS
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_b); // APPROXIMATE_HESS_KL
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_b); // NO_BACKWARD_RECONSTRUCTION
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_b); // DRAW_SOURCES_AND_RECEIVERS
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_d,
      &dummy_d1); // Q0_poroelastic,freq0_poroelastic
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // AXISYM
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // psv
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_d); // factor_subsample_image
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_b); // USE_CONSTANT_MAX_AMPLITUDE
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_d); // CONSTANT_MAX_AMPLITUDE_TO_USE
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b); // USE_SNAPSHOT_NUMBER_IN_FILENAME
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_b); // DRAW_WATER_IN_BLUE
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // US_LETTER
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_d); // POWER_DISPLAY_COLOR
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // SU_FORMAT
  specfem::fortran_IO::fortran_read_line(stream, &dummy_d); // USER_T0
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_i); // time_stepping_scheme
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_b); // ADD_PERIODIC_CONDITIONS
  specfem::fortran_IO::fortran_read_line(stream,
                                         &dummy_d); // PERIODIC_HORIZ_DIST
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // GPU_MODE
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_i); // setup_with_binary_database
  specfem::fortran_IO::fortran_read_line(stream, &dummy_i,
                                         &dummy_d), // NSTEP,DT
      specfem::fortran_IO::fortran_read_line(stream,
                                             &dummy_i); // NT_DUMP_ATTENUATION
  specfem::fortran_IO::fortran_read_line(stream, &dummy_b); // ACOUSTIC_FORCING
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_i); // NUMBER_OF_SIMULTANEOUS_RUNS
  specfem::fortran_IO::fortran_read_line(
      stream,
      &dummy_b); // BROADCAST_SAME_MESH_AND_MODEL
  specfem::fortran_IO::fortran_read_line(
      stream, &dummy_b1,
      &dummy_b2); // ADD_RANDOM_PERTURBATION_TO_THE_MESH,ADD_PERTURBATION_AROUND_SOURCE_ONLY

  mpi->sync_all();

  return std::make_tuple(nspec, npgeo, nproc);
}

specfem::kokkos::HostView2d<type_real>
specfem::mesh::IO::fortran::read_coorg_elements(std::ifstream &stream,
                                                const int npgeo,
                                                const specfem::MPI::MPI *mpi) {

  int ipoin = 0;

  type_real coorgi, coorgj;
  int buffer_length;
  specfem::kokkos::HostView2d<type_real> coorg("specfem::mesh::coorg", ndim,
                                               npgeo);

  for (int i = 0; i < npgeo; i++) {
    specfem::fortran_IO::fortran_read_line(stream, &ipoin, &coorgi, &coorgj);
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
specfem::mesh::IO::fortran::read_mesh_database_attenuation(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  int n_sls;
  type_real attenuation_f0_reference;
  bool read_velocities_at_f0;
  specfem::fortran_IO::fortran_read_line(
      stream, &n_sls, &attenuation_f0_reference, &read_velocities_at_f0);

  if (n_sls < 1) {
    throw std::runtime_error("must have N_SLS >= 1 even if attenuation if off "
                             "because it is used to assign some arrays");
  }

  return std::make_tuple(n_sls, attenuation_f0_reference,
                         read_velocities_at_f0);
}

void specfem::mesh::IO::fortran::read_mesh_database_coupled(
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
      specfem::fortran_IO::fortran_read_line(stream, &dummy_i, &dummy_i1);
  }

  if (num_fluid_poro_edges > 0) {
    for (int inum = 0; inum < num_fluid_poro_edges; inum++)
      specfem::fortran_IO::fortran_read_line(stream, &dummy_i, &dummy_i1);
  }

  if (num_solid_poro_edges > 0) {
    for (int inum = 0; inum < num_solid_poro_edges; inum++)
      specfem::fortran_IO::fortran_read_line(stream, &dummy_i, &dummy_i1);
  }
  return;
}
