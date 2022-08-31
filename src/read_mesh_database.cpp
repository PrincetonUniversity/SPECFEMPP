#include "../include/read_mesh_database.h"
#include "../include/config.h"
#include "../include/fortran_IO.h"
#include "../include/kokkos_abstractions.h"
#include "../include/mesh.h"
#include "../include/params.h"
#include "../include/read_material_properties.h"
#include "../include/specfem_mpi.h"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>

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

  return;
}

void IO::read_coorg_elements(std::ifstream &stream, specfem::mesh &mesh,
                             specfem::MPI *mpi) {

  int npgeo = mesh.npgeo, ipoin = 0;

  type_real coorgi, coorgj;
  int buffer_length;
  mesh.coorg =
      specfem::HostView2d<type_real>("specfem::mesh::coorg", ndim, npgeo);

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

  return;
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

  return;
}

void IO::read_mesh_database_mato(std::ifstream &stream, specfem::mesh &mesh,
                                 specfem::MPI *mpi) {
  int n, kmato_read, pml_read;
  std::vector<int> knods_read(mesh.properties.ngnod, -1);
  mesh.region_CPML = specfem::HostView1d<int>("specfem::mesh::region_CPML",
                                              mesh.properties.nspec);
  mesh.kmato = specfem::HostView1d<int>("specfem::mesh::region_CPML",
                                        mesh.properties.nspec);
  mesh.knods =
      specfem::HostView2d<int>("specfem::mesh::region_CPML",
                               mesh.properties.ngnod, mesh.properties.nspec);

  for (int ispec = 0; ispec < mesh.properties.nspec; ispec++) {
    mesh.kmato(ispec) = -1;
  }

  // Read an assign material values, coordinate numbering, PML association
  for (int ispec = 0; ispec < mesh.properties.nspec; ispec++) {
    // format: #element_id  #material_id #node_id1 #node_id2 #...
    IO::fortran_IO::fortran_read_line(stream, &n, &kmato_read, &knods_read,
                                      &pml_read);

    // material association
    if (n < 1 || n > mesh.properties.nspec) {
      throw std::runtime_error("Error reading mato properties");
    }
    mesh.kmato(n - 1) = kmato_read - 1;
    mesh.region_CPML(n - 1) = pml_read;

    // element control node indices
    for (int i = 0; i < mesh.properties.ngnod; i++) {
      if (knods_read[i] == -1)
        throw std::runtime_error("Error reading knods (node_id) values");

      mesh.knods(i, n) = knods_read[i];
    }
  }

  for (int ispec = 0; ispec < mesh.properties.nspec; ispec++) {
    int imat = mesh.kmato(ispec);
    if (imat < 0 || imat >= mesh.properties.numat) {
      throw std::runtime_error(
          "Error reading material properties. Invalid material ID number");
    }
  }

  return;
}

void IO::read_mesh_database_interfaces(std::ifstream &stream,
                                       specfem::interface &interface,
                                       specfem::MPI *mpi) {

  // read number of interfaces
  IO::fortran_IO::fortran_read_line(stream, &interface.ninterfaces,
                                    &interface.max_interface_size);

  mpi->cout("Number of interaces = " + std::to_string(interface.ninterfaces));

#ifdef MPI_PARALLEL
  if (interface.ninterfaces > 0) {
    interface.my_neighbors = specfem::HostView1d<int>(
        "specfem::mesh::interfaces::my_neighbors", interface.ninterfaces);
    interface.my_nelmnts_neighbors = specfem::HostView1d<int>(
        "specfem::mesh::interfaces::my_nelmnts_neighbors",
        interface.ninterfaces);
    interface.my_interfaces = specfem::HostView3d<int>(
        "specfem::mesh::interfaces::my_interfaces", interface.ninterfaces,
        interface.max_interface_size, 4);

    // initialize values
    for (int i = 0; i < interface.ninterfaces; i++) {
      interface.my_neighbors(i) = -1;
      interface.my_nelmnts_neighbors(i) = 0;
      for (int j = 0; j < interface.max_interface_size; j++) {
        for (int k = 0; k < 4; k++) {
          interface.my_interfaces(i, j, k) = -1;
        }
      }
    }
  } else {
    interface.my_neighbors =
        specfem::HostView1d<int>("specfem::mesh::interfaces::my_neighbors", 1);
    interface.my_nelmnts_neighbors = specfem::HostView1d<int>(
        "specfem::mesh::interfaces::my_nelmnts_neighbors", 1);
    interface.my_interfaces = specfem::HostView3d<int>(
        "specfem::mesh::interfaces::my_interfaces", 1, 1, 1);

    // initialize values
    interface.my_neighbors(1) = -1;
    interface.my_nelmnts_neighbors(1) = 0;
    interface.my_interfaces(1, 1, 1) = -1;
  }
#else
  if (interface.ninterfaces > 0)
    throw std::runtime_error("Found interfaces but SPECFEM compiled without "
                             "MPI. Compile SPECFEM with MPI");
  interface.my_neighbors =
      specfem::HostView1d<int>("specfem::mesh::interfaces::my_neighbors", 1);
  interface.my_nelmnts_neighbors = specfem::HostView1d<int>(
      "specfem::mesh::interfaces::my_nelmnts_neighbors", 1);
  interface.my_interfaces = specfem::HostView3d<int>(
      "specfem::mesh::interfaces::my_interfaces", 1, 1, 1);

  // initialize values
  interface.my_neighbors(1) = -1;
  interface.my_nelmnts_neighbors(1) = 0;
  interface.my_interfaces(1, 1, 1) = -1;
#endif

  // note: for serial simulations, ninterface will be zero.
  //       thus no further reading will be done below

  // reads in interfaces

#ifdef MPI_PARALLEL
  for (int num_interface = 0; num_interface < interface.ninterfaces;
       num_interface++) {
    // format: #process_interface_id  #number_of_elements_on_interface
    // where
    //     process_interface_id = rank of (neighbor) process to share MPI
    //     interface with number_of_elements_on_interface = number of interface
    //     elements
    IO::fortran_IO::fortran_read_line(
        stream, &interface.my_neighbors(num_interface),
        &interface.my_nelmnts_neighbors(num_interface));
    // loops over interface elements
    for (int ie = 0; ie < interface.my_nelmnts_neighbors(num_interface); ie++) {
      //   format: #(1)spectral_element_id  #(2)interface_type  #(3)node_id1
      //   #(4)node_id2

      //   interface types:
      //       1  -  corner point only
      //       2  -  element edge
      IO::fortran_IO::fortran_read_line(
          stream, &interface.my_interfaces(num_interface, ie, 0),
          &interface.my_interfaces(num_interface, ie, 1),
          &interface.my_interfaces(num_interface, ie, 2),
          &interface.my_interfaces(num_interface, ie, 3));
    }
  }
#endif

  return;
}

void IO::read_mesh_absorbing_boundaries(
    std::ifstream &stream, specfem::absorbing_boundary &abs_boundary,
    int &num_abs_boundary_faces, int nspec, specfem::MPI *mpi) {
  // I have to do this because std::vector<bool> is a fake container type that
  // causes issues when getting a reference
  bool codeabsread1 = true, codeabsread2 = true, codeabsread3 = true,
       codeabsread4 = true;
  std::vector<int> iedgeread(8, 0);
  int nspec_left = 0, nspec_right = 0, nspec_top = 0, nspec_bottom = 0;
  int numabsread, typeabsread;
  if (num_abs_boundary_faces < 0) {
    mpi->cout("Warning: read in negative nelemabs resetting to 0!");
    num_abs_boundary_faces = 0;
  }

  if (num_abs_boundary_faces > 0) {
    abs_boundary.numabs = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::numabs", num_abs_boundary_faces);
    abs_boundary.abs_boundary_type = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::abs_boundary_type",
        num_abs_boundary_faces);
    abs_boundary.ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1",
        num_abs_boundary_faces);
    abs_boundary.ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2",
        num_abs_boundary_faces);
    abs_boundary.ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3",
        num_abs_boundary_faces);
    abs_boundary.ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4",
        num_abs_boundary_faces);
    abs_boundary.iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1",
        num_abs_boundary_faces);
    abs_boundary.iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2",
        num_abs_boundary_faces);
    abs_boundary.iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3",
        num_abs_boundary_faces);
    abs_boundary.iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4",
        num_abs_boundary_faces);
    abs_boundary.ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", num_abs_boundary_faces);
    abs_boundary.ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", num_abs_boundary_faces);
    abs_boundary.ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", num_abs_boundary_faces);
    abs_boundary.ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", num_abs_boundary_faces);
  } else {
    abs_boundary.numabs = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::numabs", 1);
    abs_boundary.abs_boundary_type = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::abs_boundary_type", 1);
    abs_boundary.ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", 1);
    abs_boundary.ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", 1);
    abs_boundary.ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", 1);
    abs_boundary.ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", 1);
    abs_boundary.iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", 1);
    abs_boundary.iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", 1);
    abs_boundary.iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", 1);
    abs_boundary.iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", 1);
    abs_boundary.ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", 1);
    abs_boundary.ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", 1);
    abs_boundary.ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", 1);
    abs_boundary.ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", 1);
  }

  if (num_abs_boundary_faces > 0) {
    abs_boundary.codeabs =
        specfem::HostView2d<bool>("specfem::mesh::absorbing_boundary::codeabs",
                                  num_abs_boundary_faces, 4);
    abs_boundary.codeabscorner = specfem::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs_corner",
        num_abs_boundary_faces, 4);
  } else {
    abs_boundary.codeabs = specfem::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs", 1, 1);
    abs_boundary.codeabscorner = specfem::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs_corner", 1, 1);
  }

  if (num_abs_boundary_faces > 0) {
    for (int n = 0; n < num_abs_boundary_faces; n++) {
      abs_boundary.numabs(n) = 0;
      abs_boundary.abs_boundary_type(n) = 0;
      abs_boundary.ibegin_edge1(n) = 0;
      abs_boundary.ibegin_edge2(n) = 0;
      abs_boundary.ibegin_edge3(n) = 0;
      abs_boundary.ibegin_edge4(n) = 0;
      abs_boundary.iend_edge1(n) = 0;
      abs_boundary.iend_edge2(n) = 0;
      abs_boundary.iend_edge3(n) = 0;
      abs_boundary.iend_edge4(n) = 0;
      abs_boundary.ib_bottom(n) = 0;
      abs_boundary.ib_left(n) = 0;
      abs_boundary.ib_top(n) = 0;
      abs_boundary.ib_right(n) = 0;
      for (int i = 0; i < 4; i++) {
        abs_boundary.codeabs(n, i) = false;
        abs_boundary.codeabscorner(n, i) = false;
      }
    }
  } else {
    abs_boundary.numabs(1) = 0;
    abs_boundary.abs_boundary_type(1) = 0;
    abs_boundary.ibegin_edge1(1) = 0;
    abs_boundary.ibegin_edge2(1) = 0;
    abs_boundary.ibegin_edge3(1) = 0;
    abs_boundary.ibegin_edge4(1) = 0;
    abs_boundary.iend_edge1(1) = 0;
    abs_boundary.iend_edge2(1) = 0;
    abs_boundary.iend_edge3(1) = 0;
    abs_boundary.iend_edge4(1) = 0;
    abs_boundary.ib_bottom(1) = 0;
    abs_boundary.ib_left(1) = 0;
    abs_boundary.ib_top(1) = 0;
    abs_boundary.ib_right(1) = 0;
    abs_boundary.codeabs(1, 1) = false;
    abs_boundary.codeabscorner(1, 1) = false;
  }

  // user output
  mpi->cout("Todo: placeholder string - read_mesh_database.f90 line 831 - 840");

  if (num_abs_boundary_faces > 0) {
    for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
      IO::fortran_IO::fortran_read_line(
          stream, &numabsread, &codeabsread1, &codeabsread2, &codeabsread3,
          &codeabsread4, &typeabsread, &iedgeread);
      std::vector<bool> codeabsread(4, false);
      if (numabsread < 1 || numabsread > nspec)
        throw std::runtime_error("Wrong absorbing element number");
      abs_boundary.numabs(inum) = numabsread - 1;
      abs_boundary.abs_boundary_type(inum) = typeabsread;
      codeabsread[0] = codeabsread1;
      codeabsread[1] = codeabsread2;
      codeabsread[2] = codeabsread3;
      codeabsread[3] = codeabsread4;
      if (std::count(codeabsread.begin(), codeabsread.end(), true) != 1) {
        throw std::runtime_error("must have one and only one absorbing edge "
                                 "per absorbing line cited");
      }
      abs_boundary.codeabs(inum, 0) = codeabsread[0];
      abs_boundary.codeabs(inum, 1) = codeabsread[1];
      abs_boundary.codeabs(inum, 2) = codeabsread[2];
      abs_boundary.codeabs(inum, 3) = codeabsread[3];
      abs_boundary.ibegin_edge1(inum) = iedgeread[0];
      abs_boundary.iend_edge1(inum) = iedgeread[1];
      abs_boundary.ibegin_edge2(inum) = iedgeread[2];
      abs_boundary.iend_edge2(inum) = iedgeread[3];
      abs_boundary.ibegin_edge3(inum) = iedgeread[4];
      abs_boundary.iend_edge3(inum) = iedgeread[5];
      abs_boundary.ibegin_edge4(inum) = iedgeread[6];
      abs_boundary.iend_edge4(inum) = iedgeread[7];
    }

    int ncorner = 0;
    for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
      if (abs_boundary.codeabs(inum, 0)) {
        for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
             inum_duplicate++) {
          if (inum != inum_duplicate) {
            if (abs_boundary.numabs(inum) ==
                abs_boundary.numabs(inum_duplicate)) {
              if (abs_boundary.codeabs(inum_duplicate, 3)) {
                abs_boundary.codeabscorner(inum, 1) = true;
                ncorner++;
              }
              if (abs_boundary.codeabs(inum_duplicate, 1)) {
                abs_boundary.codeabscorner(inum, 2) = true;
                ncorner++;
              }
            }
          }
        }
      }
      if (abs_boundary.codeabs(inum, 2)) {
        for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
             inum_duplicate++) {
          if (inum != inum_duplicate) {
            if (abs_boundary.numabs(inum) ==
                abs_boundary.numabs(inum_duplicate)) {
              if (abs_boundary.codeabs(inum_duplicate, 3)) {
                abs_boundary.codeabscorner(inum, 3) = true;
                ncorner++;
              }
              if (abs_boundary.codeabs(inum_duplicate, 1)) {
                abs_boundary.codeabscorner(inum, 4) = true;
                ncorner++;
              }
            }
          }
        }
      }
    }

    int ncorner_all = mpi->reduce(ncorner);
    if (mpi->get_rank() == 0)
      assert(ncorner_all <= 4);

    for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
      if (abs_boundary.codeabs(inum, 0)) {
        abs_boundary.ib_bottom(inum) = nspec_bottom;
        nspec_bottom++;
      } else if (abs_boundary.codeabs(inum, 1)) {
        abs_boundary.ib_right(inum) = nspec_right;
        nspec_right++;
      } else if (abs_boundary.codeabs(inum, 2)) {
        abs_boundary.ib_top(inum) = nspec_top;
        nspec_top++;
      } else if (abs_boundary.codeabs(inum, 3)) {
        abs_boundary.ib_left(inum) = nspec_left;
        nspec_left++;
      } else {
        throw std::runtime_error(
            "incorrect absorbing boundary element type read");
      }
    }

    assert(nspec_left + nspec_right + nspec_bottom + nspec_top ==
           num_abs_boundary_faces);
  }

  mpi->cout("Todo: Placeholder string - read_mesh_database.f90 line 1092-1112");

  return;
}

void IO::read_mesh_database_acoustic_forcing(
    std::ifstream &stream, specfem::forcing_boundary &acforcing_boundary,
    int nelement_acforcing, int nspec, specfem::MPI *mpi) {
  bool codeacread1 = true, codeacread2 = true, codeacread3 = true,
       codeacread4 = true;
  std::vector<int> iedgeread(8, 0);
  int nspec_left = 0, nspec_right = 0, nspec_top = 0, nspec_bottom = 0;
  int numacread, typeacread;
  if (nelement_acforcing > 0) {
    acforcing_boundary.numacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing);
    acforcing_boundary.codeacforcing = specfem::HostView2d<bool>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing, 4);
    acforcing_boundary.typeacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing);
    acforcing_boundary.ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", nelement_acforcing);
    acforcing_boundary.ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", nelement_acforcing);
    acforcing_boundary.ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", nelement_acforcing);
    acforcing_boundary.ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", nelement_acforcing);
    acforcing_boundary.iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", nelement_acforcing);
    acforcing_boundary.iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", nelement_acforcing);
    acforcing_boundary.iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", nelement_acforcing);
    acforcing_boundary.iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", nelement_acforcing);
    acforcing_boundary.ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", nelement_acforcing);
    acforcing_boundary.ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", nelement_acforcing);
    acforcing_boundary.ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", nelement_acforcing);
    acforcing_boundary.ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", nelement_acforcing);
  } else {
    acforcing_boundary.numacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", 1);
    acforcing_boundary.codeacforcing = specfem::HostView2d<bool>(
        "specfem::mesh::forcing_boundary::numacforcing", 1, 1);
    acforcing_boundary.typeacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", 1);
    acforcing_boundary.ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", 1);
    acforcing_boundary.ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", 1);
    acforcing_boundary.ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", 1);
    acforcing_boundary.ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", 1);
    acforcing_boundary.iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", 1);
    acforcing_boundary.iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", 1);
    acforcing_boundary.iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", 1);
    acforcing_boundary.iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", 1);
    acforcing_boundary.ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", 1);
    acforcing_boundary.ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", 1);
    acforcing_boundary.ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", 1);
    acforcing_boundary.ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", 1);
  }

  if (nelement_acforcing > 0) {
    for (int n = 0; n < nelement_acforcing; n++) {
      acforcing_boundary.numacforcing(n) = 0;
      acforcing_boundary.typeacforcing(n) = 0;
      acforcing_boundary.ibegin_edge1(n) = 0;
      acforcing_boundary.ibegin_edge2(n) = 0;
      acforcing_boundary.ibegin_edge3(n) = 0;
      acforcing_boundary.ibegin_edge4(n) = 0;
      acforcing_boundary.iend_edge1(n) = 0;
      acforcing_boundary.iend_edge2(n) = 0;
      acforcing_boundary.iend_edge3(n) = 0;
      acforcing_boundary.iend_edge4(n) = 0;
      acforcing_boundary.ib_bottom(n) = 0;
      acforcing_boundary.ib_left(n) = 0;
      acforcing_boundary.ib_top(n) = 0;
      acforcing_boundary.ib_right(n) = 0;
      for (int i = 0; i < 4; i++) {
        acforcing_boundary.codeacforcing(n, i) = false;
      }
    }
  } else {
    acforcing_boundary.numacforcing(1) = 0;
    acforcing_boundary.typeacforcing(1) = 0;
    acforcing_boundary.ibegin_edge1(1) = 0;
    acforcing_boundary.ibegin_edge2(1) = 0;
    acforcing_boundary.ibegin_edge3(1) = 0;
    acforcing_boundary.ibegin_edge4(1) = 0;
    acforcing_boundary.iend_edge1(1) = 0;
    acforcing_boundary.iend_edge2(1) = 0;
    acforcing_boundary.iend_edge3(1) = 0;
    acforcing_boundary.iend_edge4(1) = 0;
    acforcing_boundary.ib_bottom(1) = 0;
    acforcing_boundary.ib_left(1) = 0;
    acforcing_boundary.ib_top(1) = 0;
    acforcing_boundary.ib_right(1) = 0;
    acforcing_boundary.codeacforcing(1, 1) = false;
  }
  if (nelement_acforcing > 0) {
    for (int inum = 0; inum < nelement_acforcing; inum++) {
      IO::fortran_IO::fortran_read_line(stream, &numacread, &codeacread1,
                                        &codeacread2, &codeacread3,
                                        &codeacread4, &typeacread, &iedgeread);
      std::vector<bool> codeacread(4, false);
      if (numacread < 1 || numacread > nspec) {
        std::runtime_error("Wrong absorbing element number");
      }
      acforcing_boundary.numacforcing(inum) = numacread - 1;
      acforcing_boundary.typeacforcing(inum) = typeacread;
      codeacread[0] = codeacread1;
      codeacread[1] = codeacread2;
      codeacread[2] = codeacread3;
      codeacread[3] = codeacread4;
      if (std::count(codeacread.begin(), codeacread.end(), true) != 1) {
        throw std::runtime_error("must have one and only one acoustic forcing "
                                 "per acoustic forcing line cited");
      }
      acforcing_boundary.codeacforcing(inum, 0) = codeacread[0];
      acforcing_boundary.codeacforcing(inum, 1) = codeacread[1];
      acforcing_boundary.codeacforcing(inum, 2) = codeacread[2];
      acforcing_boundary.codeacforcing(inum, 3) = codeacread[3];
      acforcing_boundary.ibegin_edge1(inum) = iedgeread[0];
      acforcing_boundary.iend_edge1(inum) = iedgeread[1];
      acforcing_boundary.ibegin_edge2(inum) = iedgeread[2];
      acforcing_boundary.iend_edge2(inum) = iedgeread[3];
      acforcing_boundary.ibegin_edge3(inum) = iedgeread[4];
      acforcing_boundary.iend_edge3(inum) = iedgeread[5];
      acforcing_boundary.ibegin_edge4(inum) = iedgeread[6];
      acforcing_boundary.iend_edge4(inum) = iedgeread[7];
    }

    for (int inum = 0; inum < nelement_acforcing; inum++) {
      if (acforcing_boundary.codeacforcing(inum, 0)) {
        acforcing_boundary.ib_bottom(inum) = nspec_bottom;
        nspec_bottom++;
      } else if (acforcing_boundary.codeacforcing(inum, 1)) {
        acforcing_boundary.ib_right(inum) = nspec_right;
        nspec_right++;
      } else if (acforcing_boundary.codeacforcing(inum, 2)) {
        acforcing_boundary.ib_top(inum) = nspec_top;
        nspec_top++;
      } else if (acforcing_boundary.codeacforcing(inum, 3)) {
        acforcing_boundary.ib_left(inum) = nspec_left;
        nspec_left++;
      } else {
        throw std::runtime_error(
            "Incorrect acoustic boundary element type read");
      }
    }

    assert(nspec_left + nspec_right + nspec_bottom + nspec_top ==
           nelement_acforcing);
  }

  mpi->cout("Todo: Placeholder string - read_mesh_database.f90 line 1272-1282");
}

void IO::read_mesh_database_free_surface(
    std::ifstream &stream, specfem::acoustic_free_surface &acfree_surface,
    int nelem_acoustic_surface, specfem::MPI *mpi) {

  std::vector<int> acfree_edge(4, 0);
  if (nelem_acoustic_surface > 0) {
    acfree_surface.numacfree_surface = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::numacfree_surface",
        nelem_acoustic_surface);
    acfree_surface.typeacfree_surface = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::typeacfree_surface",
        nelem_acoustic_surface);
    acfree_surface.e1 = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::e1", nelem_acoustic_surface);
    acfree_surface.e2 = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::e2", nelem_acoustic_surface);
    acfree_surface.ixmin = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::ixmin", nelem_acoustic_surface);
    acfree_surface.ixmax = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::ixmax", nelem_acoustic_surface);
    acfree_surface.izmin = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::izmin", nelem_acoustic_surface);
    acfree_surface.izmax = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::izmax", nelem_acoustic_surface);
  }

  if (nelem_acoustic_surface > 0) {
    for (int inum = 0; inum < nelem_acoustic_surface; inum++) {
      IO::fortran_IO::fortran_read_line(stream, &acfree_edge);
      acfree_surface.numacfree_surface(inum) = acfree_edge[0];
      acfree_surface.typeacfree_surface(inum) = acfree_edge[1];
      acfree_surface.e1(inum) = acfree_edge[2];
      acfree_surface.e2(inum) = acfree_edge[3];
    }
  }
}

void IO::read_mesh_database_coupled(std::ifstream &stream,
                                    int num_fluid_solid_edges,
                                    int num_fluid_poro_edges,
                                    int num_solid_poro_edges,
                                    specfem::MPI *mpi) {

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

void IO::read_mesh_database_tangential(
    std::ifstream &stream, specfem::tangential_elements &tangential_nodes,
    int nnodes_tangential_curve) {
  type_real xread, yread;
  if (nnodes_tangential_curve > 0) {
    tangential_nodes.x = specfem::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::x", nnodes_tangential_curve);
    tangential_nodes.y = specfem::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::y", nnodes_tangential_curve);
  } else {
    tangential_nodes.x =
        specfem::HostView1d<type_real>("specfem::mesh::tangential_nodes::x", 1);
    tangential_nodes.y =
        specfem::HostView1d<type_real>("specfem::mesh::tangential_nodes::y", 1);
  }

  if (nnodes_tangential_curve > 0) {
    for (int inum = 0; inum < nnodes_tangential_curve; inum++) {
      tangential_nodes.x(inum) = 0.0;
      tangential_nodes.y(inum) = 0.0;
    }
  } else {
    tangential_nodes.x(1) = 0.0;
    tangential_nodes.y(1) = 0.0;
  }

  IO::fortran_IO::fortran_read_line(stream,
                                    &tangential_nodes.force_normal_to_surface,
                                    &tangential_nodes.rec_normal_to_surface);

  if (nnodes_tangential_curve > 0) {
    for (int inum = 0; inum < nnodes_tangential_curve; inum++) {
      IO::fortran_IO::fortran_read_line(stream, &xread, &yread);
      tangential_nodes.x(inum) = xread;
      tangential_nodes.y(inum) = yread;
    }
  } else {
    tangential_nodes.force_normal_to_surface = false;
    tangential_nodes.rec_normal_to_surface = false;
  }

  return;
}

void IO::read_mesh_database_axial(std::ifstream &stream,
                                  specfem::axial_elements &axial_nodes,
                                  int nelem_on_the_axis, int nspec,
                                  specfem::MPI *mpi) {
  int ispec;
  axial_nodes.is_on_the_axis = specfem::HostView1d<bool>(
      "specfem::mesh::axial_element::is_on_the_axis", nspec);

  for (int inum = 0; inum < nspec; inum++) {
    axial_nodes.is_on_the_axis(nspec) = false;
  }

  for (int inum = 0; inum < nelem_on_the_axis; inum++) {
    IO::fortran_IO::fortran_read_line(stream, &ispec);
    if (ispec < 0 || ispec > nspec - 1)
      throw std::runtime_error(
          "ispec out of range when reading axial elements");
    axial_nodes.is_on_the_axis(ispec) = true;
  }

  return;
}

void IO::read_mesh_database(const std::string filename, specfem::mesh &mesh,
                            specfem::parameters &params,
                            std::vector<specfem::material> &materials,
                            specfem::MPI *mpi) {

  std::ifstream stream;

  stream.open(filename);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open database file");
  }

  mpi->cout("\n------------ Reading database header ----------------\n");

  try {
    IO::read_mesh_database_header(stream, mesh, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n------------ Reading global coordinates ----------------\n");
  try {
    IO::read_coorg_elements(stream, mesh, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("-- Spectral Elements --");

  int nspec_all = mpi->reduce(mesh.properties.nspec);
  int nelem_acforcing_all = mpi->reduce(mesh.properties.nelem_acforcing);
  int nelem_acoustic_surface_all =
      mpi->reduce(mesh.properties.nelem_acoustic_surface);

  std::ostringstream message;
  message << "Number of spectral elements . . . . . . . . . .(nspec) = "
          << nspec_all
          << "\n"
             "Number of control nodes per element . . . . . .(NGNOD) = "
          << mesh.properties.ngnod
          << "\n"
             "Number of points for display . . . . . . .(pointsdisp) = "
          << mesh.properties.pointsdisp
          << "\n"
             "Number of element material sets . . . . . . . .(numat) = "
          << mesh.properties.numat
          << "\n"
             "Number of acoustic forcing elements .(nelem_acforcing) = "
          << nelem_acforcing_all
          << "\n"
             "Number of acoustic free surf .(nelem_acoustic_surface) = "
          << nelem_acoustic_surface_all;

  mpi->cout(message.str());

  mpi->cout("\n------------ Reading database attenuation ----------------\n");

  try {
    IO::read_mesh_database_attenuation(stream, params, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n------------ Reading material properties ----------------\n");

  try {
    materials =
        IO::read_material_properties(stream, mesh.properties.numat, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout(
      "\n------------ Reading material specifications ----------------\n");

  try {
    IO::read_mesh_database_mato(stream, mesh, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout(
      "\n------- Reading MPI interfaces for allocating MPI buffers --------\n");

  try {
    IO::read_mesh_database_interfaces(stream, mesh.inter, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n------------ Reading absorbing boundaries --------------\n");

  try {
    IO::read_mesh_absorbing_boundaries(stream, mesh.abs_boundary,
                                       mesh.properties.nelemabs,
                                       mesh.properties.nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout(
      "\n-------------- Reading acoustic forcing boundary-----------------\n");

  try {
    IO::read_mesh_database_acoustic_forcing(stream, mesh.acforcing_boundary,
                                            mesh.properties.nelem_acforcing,
                                            mesh.properties.nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n-------------- Reading acoustic free surface--------------\n");

  try {
    IO::read_mesh_database_free_surface(stream, mesh.acfree_surface,
                                        mesh.properties.nelem_acoustic_surface,
                                        mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n--------------------Read mesh coupled elements--------------\n");

  mpi->cout("\n********** Coupled interfaces have not been impletmented yet "
            "**********\n");

  try {
    IO::read_mesh_database_coupled(stream,
                                   mesh.properties.num_fluid_solid_edges,
                                   mesh.properties.num_fluid_poro_edges,
                                   mesh.properties.num_solid_poro_edges, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n-------------Read mesh tangential elements---------------\n");

  try {
    IO::read_mesh_database_tangential(stream, mesh.tangential_nodes,
                                      mesh.properties.nnodes_tagential_curve);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n----------------Read mesh axial elements-----------------\n");

  try {
    IO::read_mesh_database_axial(stream, mesh.axial_nodes,
                                 mesh.properties.nelem_on_the_axis, mesh.nspec,
                                 mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // Check if database file was read completely
  if (stream.get() && !stream.eof()) {
    throw std::runtime_error("The Database file wasn't fully read. Is there "
                             "anything written after axial elements?");
  }

  stream.close();
}
