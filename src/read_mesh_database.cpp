#include "../include/read_mesh_database.h"
#include "../include/boundaries.h"
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
#include <tuple>

// Find corner elements of the absorbing boundary
void find_corners(const specfem::HostView1d<int> numabs,
                  const specfem::HostView2d<bool> codeabs,
                  specfem::HostView2d<bool> codeabscorner,
                  const int num_abs_boundary_faces, specfem::MPI *mpi) {
  int ncorner = 0;
  for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
    if (codeabs(inum, 0)) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (numabs(inum) == numabs(inum_duplicate)) {
            if (codeabs(inum_duplicate, 3)) {
              codeabscorner(inum, 1) = true;
              ncorner++;
            }
            if (codeabs(inum_duplicate, 1)) {
              codeabscorner(inum, 2) = true;
              ncorner++;
            }
          }
        }
      }
    }
    if (codeabs(inum, 2)) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (numabs(inum) == numabs(inum_duplicate)) {
            if (codeabs(inum_duplicate, 3)) {
              codeabscorner(inum, 3) = true;
              ncorner++;
            }
            if (codeabs(inum_duplicate, 1)) {
              codeabscorner(inum, 4) = true;
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
}

void calculate_ib(const specfem::HostView2d<bool> code,
                  specfem::HostView1d<int> ib_bottom,
                  specfem::HostView1d<int> ib_top,
                  specfem::HostView1d<int> ib_left,
                  specfem::HostView1d<int> ib_right, const int nelements) {

  int nspec_left = 0, nspec_right = 0, nspec_top = 0, nspec_bottom = 0;
  for (int inum = 0; inum < nelements; inum++) {
    if (code(inum, 0)) {
      ib_bottom(inum) = nspec_bottom;
      nspec_bottom++;
    } else if (code(inum, 1)) {
      ib_right(inum) = nspec_right;
      nspec_right++;
    } else if (code(inum, 2)) {
      ib_top(inum) = nspec_top;
      nspec_top++;
    } else if (code(inum, 3)) {
      ib_left(inum) = nspec_left;
      nspec_left++;
    } else {
      throw std::runtime_error("Incorrect acoustic boundary element type read");
    }
  }

  assert(nspec_left + nspec_right + nspec_bottom + nspec_top == nelements);
}

std::tuple<int, int, int> IO::read_mesh_database_header(std::ifstream &stream,
                                                        specfem::MPI *mpi) {
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

specfem::HostView2d<type_real> IO::read_coorg_elements(std::ifstream &stream,
                                                       const int npgeo,
                                                       specfem::MPI *mpi) {

  int ipoin = 0;

  type_real coorgi, coorgj;
  int buffer_length;
  specfem::HostView2d<type_real> coorg("specfem::mesh::coorg", ndim, npgeo);

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

specfem::properties IO::read_mesh_properties(std::ifstream &stream,
                                             specfem::MPI *mpi) {
  // ---------------------------------------------------------------------
  // reading mesh properties
  specfem::properties properties;

  IO::fortran_IO::fortran_read_line(
      stream, &properties.numat, &properties.ngnod, &properties.nspec,
      &properties.pointsdisp, &properties.plot_lowerleft_corner_only);

  IO::fortran_IO::fortran_read_line(
      stream, &properties.nelemabs, &properties.nelem_acforcing,
      &properties.nelem_acoustic_surface, &properties.num_fluid_solid_edges,
      &properties.num_fluid_poro_edges, &properties.num_solid_poro_edges,
      &properties.nnodes_tangential_curve, &properties.nelem_on_the_axis);
  // ----------------------------------------------------------------------

  mpi->sync_all();
  return properties;
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

specfem::materials::material_ind
IO::read_mesh_database_mato(std::ifstream &stream, const int ngnod,
                            const int nspec, const int numat,
                            specfem::MPI *mpi) {
  std::vector<int> knods_read(ngnod, -1);
  specfem::materials::material_ind material_indic(nspec, ngnod);
  int n, kmato_read, pml_read;
  // Read an assign material values, coordinate numbering, PML association
  for (int ispec = 0; ispec < nspec; ispec++) {
    // format: #element_id  #material_id #node_id1 #node_id2 #...
    IO::fortran_IO::fortran_read_line(stream, &n, &kmato_read, &knods_read,
                                      &pml_read);

    // material association
    if (n < 1 || n > nspec) {
      throw std::runtime_error("Error reading mato properties");
    }
    material_indic.kmato(n - 1) = kmato_read - 1;
    material_indic.region_CPML(n - 1) = pml_read;

    // element control node indices (ipgeo)
    for (int i = 0; i < ngnod; i++) {
      if (knods_read[i] == 0)
        throw std::runtime_error("Error reading knods (node_id) values");

      material_indic.knods(i, n - 1) = knods_read[i] - 1;
    }
  }

  for (int ispec = 0; ispec < nspec; ispec++) {
    int imat = material_indic.kmato(ispec);
    if (imat < 0 || imat >= numat) {
      throw std::runtime_error(
          "Error reading material properties. Invalid material ID number");
    }
  }

  return material_indic;
}

specfem::interfaces::interface IO::read_mesh_database_interfaces(
    std::ifstream &stream, specfem::MPI *mpi) {

  // read number of interfaces
  // Where these 2 values are written needs to change in new database format
  int ninterfaces, max_interface_size;
  IO::fortran_IO::fortran_read_line(stream, &ninterfaces, &max_interface_size);

  mpi->cout("Number of interaces = " + std::to_string(ninterfaces));

  // allocate interface variables
  specfem::interfaces::interface interface(ninterfaces, max_interface_size);

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

  return interface;
}

specfem::boundaries::absorbing_boundary
IO::read_mesh_absorbing_boundaries(std::ifstream &stream,
                                   int num_abs_boundary_faces, int nspec,
                                   specfem::MPI *mpi) {

  // I have to do this because std::vector<bool> is a fake container type that
  // causes issues when getting a reference
  bool codeabsread1 = true, codeabsread2 = true, codeabsread3 = true,
       codeabsread4 = true;
  std::vector<int> iedgeread(8, 0);
  int numabsread, typeabsread;
  if (num_abs_boundary_faces < 0) {
    mpi->cout("Warning: read in negative nelemabs resetting to 0!");
    num_abs_boundary_faces = 0;
  }

  specfem::boundaries::absorbing_boundary abs_boundary(num_abs_boundary_faces);

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

    // Find corner elements
    find_corners(abs_boundary.numabs, abs_boundary.codeabs,
                 abs_boundary.codeabscorner, num_abs_boundary_faces, mpi);

    // populate ib_bottom, ib_top, ib_left, ib_right arrays
    calculate_ib(abs_boundary.codeabs, abs_boundary.ib_bottom,
                 abs_boundary.ib_top, abs_boundary.ib_left,
                 abs_boundary.ib_right, num_abs_boundary_faces);
  }

  mpi->cout("Todo: Placeholder string - read_mesh_database.f90 line 1092-1112");

  return abs_boundary;
}

specfem::boundaries::forcing_boundary
IO::read_mesh_database_acoustic_forcing(std::ifstream &stream,
                                        int nelement_acforcing, int nspec,
                                        specfem::MPI *mpi) {
  bool codeacread1 = true, codeacread2 = true, codeacread3 = true,
       codeacread4 = true;
  std::vector<int> iedgeread(8, 0);
  int numacread, typeacread;

  specfem::boundaries::forcing_boundary acforcing_boundary(nelement_acforcing);

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
  }

  // populate ib_bottom, ib_top, ib_left, ib_right arrays
  calculate_ib(acforcing_boundary.codeacforcing, acforcing_boundary.ib_bottom,
               acforcing_boundary.ib_top, acforcing_boundary.ib_left,
               acforcing_boundary.ib_right, nelement_acforcing);

  mpi->cout("Todo: Placeholder string - read_mesh_database.f90 line 1272-1282");

  return acforcing_boundary;
}

specfem::surfaces::acoustic_free_surface IO::read_mesh_database_free_surface(
    std::ifstream &stream, int nelem_acoustic_surface, specfem::MPI *mpi) {

  std::vector<int> acfree_edge(4, 0);
  specfem::surfaces::acoustic_free_surface acfree_surface(
      nelem_acoustic_surface);

  if (nelem_acoustic_surface > 0) {
    for (int inum = 0; inum < nelem_acoustic_surface; inum++) {
      IO::fortran_IO::fortran_read_line(stream, &acfree_edge);
      acfree_surface.numacfree_surface(inum) = acfree_edge[0];
      acfree_surface.typeacfree_surface(inum) = acfree_edge[1];
      acfree_surface.e1(inum) = acfree_edge[2];
      acfree_surface.e2(inum) = acfree_edge[3];
    }
  }

  return acfree_surface;
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

specfem::elements::tangential_elements
IO::read_mesh_database_tangential(std::ifstream &stream,
                                  int nnodes_tangential_curve) {
  type_real xread, yread;

  specfem::elements::tangential_elements tangential_nodes(
      nnodes_tangential_curve);

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

  return tangential_nodes;
}

specfem::elements::axial_elements
IO::read_mesh_database_axial(std::ifstream &stream, int nelem_on_the_axis,
                             int nspec, specfem::MPI *mpi) {
  int ispec;

  specfem::elements::axial_elements axial_nodes(nspec);
  for (int inum = 0; inum < nelem_on_the_axis; inum++) {
    IO::fortran_IO::fortran_read_line(stream, &ispec);
    if (ispec < 0 || ispec > nspec - 1)
      throw std::runtime_error(
          "ispec out of range when reading axial elements");
    axial_nodes.is_on_the_axis(ispec) = true;
  }

  return axial_nodes;
}

std::tuple<specfem::mesh, std::vector<specfem::material *> >
IO::read_mesh_database(const std::string filename, specfem::parameters &params,
                       specfem::MPI *mpi) {

  // Driver function to read database file
  std::ifstream stream;

  specfem::mesh mesh;
  std::vector<specfem::material *> materials;

  stream.open(filename);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open database file");
  }

  mpi->cout("\n------------ Reading database header ----------------\n");

  try {
    auto [nspec, npgeo, nproc] = IO::read_mesh_database_header(stream, mpi);
    mesh.nspec = nspec;
    mesh.npgeo = npgeo;
    mesh.nproc = nproc;
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n------------ Reading global coordinates ----------------\n");
  try {
    mesh.coorg = IO::read_coorg_elements(stream, mesh.npgeo, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n--------------Reading mesh properties---------------\n");

  try {
    mesh.parameters = IO::read_mesh_properties(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("-- Spectral Elements --");

  int nspec_all = mpi->reduce(mesh.parameters.nspec);
  int nelem_acforcing_all = mpi->reduce(mesh.parameters.nelem_acforcing);
  int nelem_acoustic_surface_all =
      mpi->reduce(mesh.parameters.nelem_acoustic_surface);

  std::ostringstream message;
  message << "Number of spectral elements . . . . . . . . . .(nspec) = "
          << nspec_all
          << "\n"
             "Number of control nodes per element . . . . . .(NGNOD) = "
          << mesh.parameters.ngnod
          << "\n"
             "Number of points for display . . . . . . .(pointsdisp) = "
          << mesh.parameters.pointsdisp
          << "\n"
             "Number of element material sets . . . . . . . .(numat) = "
          << mesh.parameters.numat
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
        IO::read_material_properties(stream, mesh.parameters.numat, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout(
      "\n------------ Reading material specifications ----------------\n");

  try {
    mesh.material_ind = IO::read_mesh_database_mato(
        stream, mesh.parameters.ngnod, mesh.nspec, mesh.parameters.numat, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout(
      "\n------- Reading MPI interfaces for allocating MPI buffers --------\n");

  try {
    mesh.interface = IO::read_mesh_database_interfaces(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n------------ Reading absorbing boundaries --------------\n");

  try {
    mesh.abs_boundary = IO::read_mesh_absorbing_boundaries(
        stream, mesh.parameters.nelemabs, mesh.parameters.nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout(
      "\n-------------- Reading acoustic forcing boundary-----------------\n");

  try {
    mesh.acforcing_boundary = IO::read_mesh_database_acoustic_forcing(
        stream, mesh.parameters.nelem_acforcing, mesh.parameters.nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n-------------- Reading acoustic free surface--------------\n");

  try {
    mesh.acfree_surface = IO::read_mesh_database_free_surface(
        stream, mesh.parameters.nelem_acoustic_surface, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n--------------------Read mesh coupled elements--------------\n");

  mpi->cout("\n********** Coupled interfaces have not been impletmented yet "
            "**********\n");

  try {
    IO::read_mesh_database_coupled(stream,
                                   mesh.parameters.num_fluid_solid_edges,
                                   mesh.parameters.num_fluid_poro_edges,
                                   mesh.parameters.num_solid_poro_edges, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n-------------Read mesh tangential elements---------------\n");

  try {
    mesh.tangential_nodes = IO::read_mesh_database_tangential(
        stream, mesh.parameters.nnodes_tangential_curve);
  } catch (std::runtime_error &e) {
    throw;
  }

  mpi->cout("\n----------------Read mesh axial elements-----------------\n");

  try {
    mesh.axial_nodes = IO::read_mesh_database_axial(
        stream, mesh.parameters.nelem_on_the_axis, mesh.nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // Check if database file was read completely
  if (stream.get() && !stream.eof()) {
    throw std::runtime_error("The Database file wasn't fully read. Is there "
                             "anything written after axial elements?");
  }

  stream.close();

  return std::make_tuple(mesh, materials);
}
