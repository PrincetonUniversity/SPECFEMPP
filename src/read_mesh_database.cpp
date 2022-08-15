#include "../include/read_mesh_database.h"
#include "../include/config.h"
#include "../include/fortran_IO.h"
#include "../include/kokkos_abstractions.h"
#include "../include/mesh.h"
#include "../include/params.h"
#include "../include/specfem_mpi.h"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>

using HostView2d = specfem::HostView2d<type_real>;

void IO::read_mesh_database_header(std::ifstream &stream, specfem::mesh &mesh,
                                   specfem::MPI *mpi) {
  // This subroutine reads header values of the database which are skipped
  std::string dummy_s;
  int dummy_i, dummy_i1, dummy_i2;
  type_real dummy_d, dummy_d1;
  bool dummy_b, dummy_b1, dummy_b2, dummy_b3;

  IO::fortran_IO::fortran_read_line(stream, &dummy_s); // title
  IO::fortran_IO::fortran_read_line(stream, &dummy_i,
                                    &dummy_b); // noise tomography,
                                               // undo_attenuation_and_or_PML
  IO::fortran_IO::fortran_read_line(stream, &mesh.nspec); // nspec
  IO::fortran_IO::fortran_read_line(stream, &mesh.npgeo,
                                    &mesh.nproc); // ngeo, nproc
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
}

void IO::read_coorg_elements(std::ifstream &stream, specfem::mesh &mesh,
                             specfem::MPI *mpi) {

  int npgeo = mesh.npgeo, ipoin = 0;

  type_real coorgi, coorgj;
  int buffer_length;
  mesh.coorg = HostView2d("specfem::mesh::coorg", ndim, npgeo);
  for (int i = 0; i < npgeo; i++) {
    IO::fortran_IO::fortran_read_line(stream, &ipoin, &coorgi, &coorgj);
    if (ipoin < 1 || ipoin > npgeo) {
      throw std::runtime_error("Error reading coordinates");
    }
    mesh.coorg(0, ipoin - 1) = coorgi;
    mesh.coorg(1, ipoin - 1) = coorgj;
  }

  specfem::prop &properties = mesh.properties;

  IO::fortran_IO::fortran_read_line(
      stream, &properties.numat, &properties.ngnod, &properties.nspec,
      &properties.pointsdisp, &properties.plot_lowerleft_corner_only);

  IO::fortran_IO::fortran_read_line(
      stream, &properties.nelemabs, &properties.nelem_acforcing,
      &properties.nelem_acoustic_surface, &properties.num_fluid_solid_edges,
      &properties.num_fluid_poro_edges, &properties.num_solid_poro_edges,
      &properties.nnodes_tagential_curve, &properties.nelem_on_the_axis);
}

void IO::read_mesh_database_attenuation(std::ifstream &stream,
                                        specfem::parameters &params,
                                        specfem::MPI *mpi) {

  IO::fortran_IO::fortran_read_line(stream, &params.n_sls,
                                    &params.attenuation_f0_reference,
                                    &params.read_velocities_at_f0);

  if (params.n_sls < 1) {
    throw std::runtime_error("must have N_SLS >= 1 even if attenuation if off "
                             "because it is used to assign some arrays");
  }
}

// void read_mesh_databases_mato(std::ifstream &stream,
//                                 specfem::mesh &mesh, specfem::MPI *mpi){

//   std::vector<type_real> knods_read(mesh.properties.ngnod, 0);
//   using HostView1d = HostView1d<type_real>;

// }
