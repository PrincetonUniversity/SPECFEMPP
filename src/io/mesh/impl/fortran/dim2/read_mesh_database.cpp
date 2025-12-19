#include "io/mesh/impl/fortran/dim2/read_mesh_database.hpp"
#include "io/fortranio/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/mpi.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <tuple>

std::tuple<int, int, int>
specfem::io::mesh::impl::fortran::dim2::read_mesh_database_header(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {
  // This subroutine reads header values of the database which are skipped
  std::string dummy_s;
  int dummy_i, dummy_i1, dummy_i2;
  double dummy_d, dummy_d1; // there is no type_real in the meshfem fortran code
  bool dummy_b, dummy_b1, dummy_b2, dummy_b3;
  int nspec, npgeo, nproc;

  specfem::io::fortran_read_line(stream, &dummy_s); // title
  specfem::io::fortran_read_line(stream, &dummy_i,
                                 &dummy_b);       // noise tomography,
                                                  // undo_attenuation_and_or_PML
  specfem::io::fortran_read_line(stream, &nspec); // nspec
  specfem::io::fortran_read_line(stream, &npgeo,
                                 &nproc); // ngeo, nproc
  specfem::io::fortran_read_line(stream, &dummy_b1,
                                 &dummy_b2); // output_grid_per_plot,
                                             // interpol
  specfem::io::fortran_read_line(stream,
                                 &dummy_i); // ntstep_between_output_info
  specfem::io::fortran_read_line(stream,
                                 &dummy_i); // ntstep_between_output_seismos
  specfem::io::fortran_read_line(stream,
                                 &dummy_i); // ntstep_between_output_images
  specfem::io::fortran_read_line(stream, &dummy_b); // PML-boundary
  specfem::io::fortran_read_line(stream,
                                 &dummy_b);         // rotate_PML_activate
  specfem::io::fortran_read_line(stream, &dummy_d); // rotate_PML_angle
  specfem::io::fortran_read_line(stream, &dummy_d); // k_min_pml
  specfem::io::fortran_read_line(stream, &dummy_d); // m_max_PML
  specfem::io::fortran_read_line(stream,
                                 &dummy_d); // damping_change_factor_acoustic
  specfem::io::fortran_read_line(stream,
                                 &dummy_d); // damping_change_factor_elastic
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // PML_parameter_adjustment
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // read_external_mesh
  specfem::io::fortran_read_line(stream,
                                 &dummy_i); // nelem_PML_thickness
  specfem::io::fortran_read_line(
      stream, &dummy_i, &dummy_i1,
      &dummy_i2); // NTSTEP_BETWEEN_OUTPUT_SAMPLE,imagetype_JPEG,imagetype_wavefield_dumps
  specfem::io::fortran_read_line(
      stream, &dummy_b1,
      &dummy_b2); // output_postscript_snapshot,output_color_image
  specfem::io::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_b2, &dummy_d, &dummy_i1,
      &dummy_d1); // meshvect,modelvect,boundvect,cutsnaps,subsamp_postscript,sizemax_arrows
  specfem::io::fortran_read_line(stream, &dummy_d); // anglerec
  specfem::io::fortran_read_line(stream, &dummy_b); // initialfield
  specfem::io::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_b2,
      &dummy_b3); // add_Bielak_conditions_bottom,add_Bielak_conditions_right,add_Bielak_conditions_top,add_Bielak_conditions_left
  specfem::io::fortran_read_line(stream, &dummy_s,
                                 &dummy_i); // seismotype,imagetype_postscript
  specfem::io::fortran_read_line(stream, &dummy_s); // model
  specfem::io::fortran_read_line(stream, &dummy_s); // save_model
  specfem::io::fortran_read_line(stream, &dummy_s); // Tomography file
  specfem::io::fortran_read_line(
      stream, &dummy_b, &dummy_b1, &dummy_i,
      &dummy_b2); // output_grid_ASCII,OUTPUT_ENERGY,NTSTEP_BETWEEN_OUTPUT_ENERGY,output_wavefield_dumps
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // use_binary_for_wavefield_dumps
  specfem::io::fortran_read_line(
      stream, &dummy_b, &dummy_b1,
      &dummy_b2); // ATTENUATION_VISCOELASTIC,ATTENUATION_PORO_FLUID_PART,ATTENUATION_VISCOACOUSTIC
  specfem::io::fortran_read_line(stream, &dummy_b); // use_solvopt
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // save_ASCII_seismograms
  specfem::io::fortran_read_line(
      stream, &dummy_b,
      &dummy_b1); // save_binary_seismograms_single,save_binary_seismograms_double
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // USE_TRICK_FOR_BETTER_PRESSURE
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // COMPUTE_INTEGRATED_ENERGY_FIELD
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // save_ASCII_kernels
  specfem::io::fortran_read_line(stream,
                                 &dummy_i); // NTSTEP_BETWEEN_COMPUTE_KERNELS
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // APPROXIMATE_HESS_KL
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // NO_BACKWARD_RECONSTRUCTION
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // DRAW_SOURCES_AND_RECEIVERS
  specfem::io::fortran_read_line(stream, &dummy_d,
                                 &dummy_d1); // Q0_poroelastic,freq0_poroelastic

  // TODO: This is a hack to skip the attenuation parameters temporarily
  //       This should be fixed in the future when the attenuation is
  //       implemented or just after the Hackathon with Christina.
  // Store the current position in the stream so we can go back to it
  std::streampos savedPosition = stream.tellg();
  try {
    specfem::io::fortran_read_line(stream, &dummy_b, &dummy_b1,
                                   &dummy_d1); // ATTENUATION_PERMITTIVITY,
                                               // ATTENUATION_CONDUCTIVITY,
                                               // f0_electromagnetic
  } catch (const std::exception &e) {
    // Go back to initial position
    stream.seekg(savedPosition);
  };
  specfem::io::fortran_read_line(stream, &dummy_b); // AXISYM
  specfem::io::fortran_read_line(stream, &dummy_b); // psv
  specfem::io::fortran_read_line(stream,
                                 &dummy_d); // factor_subsample_image
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // USE_CONSTANT_MAX_AMPLITUDE
  specfem::io::fortran_read_line(stream,
                                 &dummy_d); // CONSTANT_MAX_AMPLITUDE_TO_USE
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // USE_SNAPSHOT_NUMBER_IN_FILENAME
  specfem::io::fortran_read_line(stream,
                                 &dummy_b);         // DRAW_WATER_IN_BLUE
  specfem::io::fortran_read_line(stream, &dummy_b); // US_LETTER
  specfem::io::fortran_read_line(stream,
                                 &dummy_d);         // POWER_DISPLAY_COLOR
  specfem::io::fortran_read_line(stream, &dummy_b); // SU_FORMAT
  specfem::io::fortran_read_line(stream, &dummy_d); // USER_T0
  specfem::io::fortran_read_line(stream,
                                 &dummy_i); // time_stepping_scheme
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // ADD_PERIODIC_CONDITIONS
  specfem::io::fortran_read_line(stream,
                                 &dummy_d);         // PERIODIC_HORIZ_DIST
  specfem::io::fortran_read_line(stream, &dummy_b); // GPU_MODE
  specfem::io::fortran_read_line(stream,
                                 &dummy_i); // setup_with_binary_database
  specfem::io::fortran_read_line(stream, &dummy_i,
                                 &dummy_d), // NSTEP,DT
      specfem::io::fortran_read_line(stream,
                                     &dummy_i);     // NT_DUMP_ATTENUATION
  specfem::io::fortran_read_line(stream, &dummy_b); // ACOUSTIC_FORCING
  specfem::io::fortran_read_line(stream,
                                 &dummy_i); // NUMBER_OF_SIMULTANEOUS_RUNS
  specfem::io::fortran_read_line(stream,
                                 &dummy_b); // BROADCAST_SAME_MESH_AND_MODEL
  specfem::io::fortran_read_line(
      stream, &dummy_b1,
      &dummy_b2); // ADD_RANDOM_PERTURBATION_TO_THE_MESH,ADD_PERTURBATION_AROUND_SOURCE_ONLY

  specfem::MPI_new::sync_all();

  return std::make_tuple(nspec, npgeo, nproc);
}

specfem::kokkos::HostView2d<type_real>
specfem::io::mesh::impl::fortran::dim2::read_coorg_elements(
    std::ifstream &stream, const int npgeo, const specfem::MPI::MPI *mpi) {

  int ipoin = 0;

  double coorgi, coorgj;
  specfem::kokkos::HostView2d<type_real> coorg("specfem::mesh::coorg", ndim,
                                               npgeo);

  for (int i = 0; i < npgeo; i++) {
    specfem::io::fortran_read_line(stream, &ipoin, &coorgi, &coorgj);
    if (ipoin < 1 || ipoin > npgeo) {
      throw std::runtime_error("Error reading coordinates");
    }
    // coorg stores the x,z for every control point
    // coorg([0, 2), i) = [x, z]
    coorg(0, ipoin - 1) = static_cast<type_real>(coorgi);
    coorg(1, ipoin - 1) = static_cast<type_real>(coorgj);
  }

  return coorg;
}

std::tuple<int, type_real, bool>
specfem::io::mesh::impl::fortran::dim2::read_mesh_database_attenuation(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  int n_sls;
  double attenuation_f0_reference;
  bool read_velocities_at_f0;
  specfem::io::fortran_read_line(stream, &n_sls, &attenuation_f0_reference,
                                 &read_velocities_at_f0);

  //   if (n_sls < 1) {
  //     throw std::runtime_error("must have N_SLS >= 1 even if attenuation if
  //     off "
  //                              "because it is used to assign some arrays");
  //   }

  return std::make_tuple(n_sls,
                         static_cast<type_real>(attenuation_f0_reference),
                         read_velocities_at_f0);
}
