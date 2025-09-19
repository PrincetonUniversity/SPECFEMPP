// Internal Includes
#include "mesh/mesh.hpp"
#include "enumerations/interface.hpp"
#include "io/fortranio/interface.hpp"
#include "io/interface.hpp"
#include "io/mesh/impl/fortran/dim3/interface.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"

// External/Standard Libraries
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

template <specfem::element::medium_tag medium>
std::string try_print_medium_element(
    const specfem::mesh::element_types<specfem::dimension::type::dim3>
        &element_types,
    int index) {

  std::ostringstream message;
  try {
    message << element_types.print<medium>(index);
  } catch (std::runtime_error &e) {
    message << e.what();
  } catch (...) {
    message << "Unknown exception caught in try_print_medium_element"
            << ".\n";
  }
  return message.str();
}

specfem::mesh::mesh<specfem::dimension::type::dim3>
specfem::io::read_3d_mesh(const std::string &mesh_parameters_file,
                          const std::string &mesh_databases_file,
                          const specfem::MPI::MPI *mpi) {

  // Creating aliases for Checking functions
  using specfem::io::mesh::impl::fortran::dim3::check_read_test_value;
  using specfem::io::mesh::impl::fortran::dim3::check_values;

  // Creating aliases for Array Reading functions
  using specfem::io::mesh::impl::fortran::dim3::try_read_array;
  using specfem::io::mesh::impl::fortran::dim3::
      try_read_control_nodes_coordinates;
  using specfem::io::mesh::impl::fortran::dim3::try_read_control_nodes_indexing;
  using specfem::io::mesh::impl::fortran::dim3::try_read_index_array;
  using specfem::io::mesh::impl::fortran::dim3::try_read_line;

  // Declaring empty mesh objects
  specfem::mesh::mesh<specfem::dimension::type::dim3> mesh;

  // Open the database file
  std::ifstream stream;
  stream.open(mesh_parameters_file);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open mesh parameter file");
  }

  try {
    mesh.parameters =
        specfem::io::mesh::impl::fortran::dim3::read_mesh_parameters(stream,
                                                                     mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // Mesh Parameters are populated, closing the parameters file.
  stream.close();

#ifndef NDEBUG
  // Print the parameters
  mpi->cout(mesh.parameters.print());
#endif

  // Open the database file
  stream.open(mesh_databases_file);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open mesh database file");
  }

  int nspec, nglob, nspec_irregular, ngnod;

  try_read_line("read_nspec", stream, &nspec);
  try_read_line("read_nglob", stream, &nglob);
  try_read_line("read_nspec_irregular", stream, &ngnod);
  try_read_line("read_nspec_irregular", stream, &nspec_irregular);

  // Check values
  check_values("nspec", nspec, mesh.parameters.nspec);
  check_values("nglob", nglob, mesh.parameters.nglob);
  check_values("ngnod", ngnod, mesh.parameters.ngnod);
  check_values("nspec_irregular", nspec_irregular,
               mesh.parameters.nspec_irregular);

  // Create the mapping object
  mesh.mapping = specfem::mesh::mapping<specfem::dimension::type::dim3>(
      mesh.parameters.nspec, mesh.parameters.nglob,
      mesh.parameters.nspec_irregular, mesh.parameters.ngllx,
      mesh.parameters.nglly, mesh.parameters.ngllz);

  // Reading the mapping from the database file.
  try_read_index_array("read_ibool", stream, mesh.mapping.ibool);

#ifndef NDEBUG
  // Print Mapping parameters and the first spectral element
  mpi->cout(mesh.mapping.print());
  mpi->cout(mesh.mapping.print(0));
  mpi->cout(mesh.mapping.print(mesh.parameters.nspec - 1));
#endif

  // Create the coordinates object
  mesh.coordinates = specfem::mesh::coordinates<specfem::dimension::type::dim3>(
      mesh.parameters.nspec, mesh.parameters.nglob, mesh.parameters.ngllx,
      mesh.parameters.nglly, mesh.parameters.ngllz);

  // Reading the coordinates from the database file.
  try_read_array("read_x", stream, mesh.coordinates.x);
  try_read_array("read_y", stream, mesh.coordinates.y);
  try_read_array("read_z", stream, mesh.coordinates.z);

#ifndef NDEBUG
  // Print Coordinates parameters and the first global node
  mpi->cout(mesh.coordinates.print());
  mpi->cout(mesh.coordinates.print(0));
  mpi->cout(mesh.coordinates.print(mesh.parameters.nglob - 1));

  // These print the coordinate array layout for the first spectral element
  // for debugging the array layout (Fortran v. C)
  // mpi->cout(mesh.coordinates.print(0, mesh.mapping, "x"));
  // mpi->cout(mesh.coordinates.print(0, mesh.mapping, "y"));
  // mpi->cout(mesh.coordinates.print(0, mesh.mapping, "z"));
#endif

  // Initialize irregular element number array
  mesh.irregular_element_number = decltype(mesh.irregular_element_number)(
      "irregular_element_number", mesh.parameters.nspec);

  // Read Irregular elements
  try_read_index_array("read_irregular_element_number", stream,
                       mesh.irregular_element_number);

  // Read the Jacobian matrix (only two CUSTOM_REALs)
  try_read_line("read_xix_regular", stream, &mesh.xix_regular);
  try_read_line("read_jacobian_regular", stream, &mesh.jacobian_regular);

#ifndef NDEBUG
  // Print the first and last irregular element
  std::ostringstream message;
  message << "First irregular element: " << mesh.irregular_element_number(0)
          << "\n";
  message << "Last irregular element: "
          << mesh.irregular_element_number(mesh.parameters.nspec_irregular - 1)
          << "\n";

  // Print xix and jacobian
  message << "xix_regular: " << mesh.xix_regular << "\n";
  message << "jacobian_regular: " << mesh.jacobian_regular << "\n";
  mpi->cout(message.str());
#endif

  // Fix for reading a regular mesh with 0 irregular elements
  int nspec_irregular_in, ngllx_in, nglly_in, ngllz_in;
  if (mesh.parameters.nspec_irregular == 0) {
    nspec_irregular_in = 1;
    ngllx_in = 1;
    nglly_in = 1;
    ngllz_in = 1;
  } else {
    nspec_irregular_in = mesh.parameters.nspec_irregular;
    ngllx_in = mesh.parameters.ngllx;
    nglly_in = mesh.parameters.nglly;
    ngllz_in = mesh.parameters.ngllz;
  }
  // Create the Jacobian matrix object
  mesh.jacobian_matrix =
      specfem::mesh::jacobian_matrix<specfem::dimension::type::dim3>(
          nspec_irregular_in, ngllx_in, nglly_in, ngllz_in);

  // Adding already read parameters for the jacobian_matrix
  mesh.jacobian_matrix.nspec_irregular = nspec_irregular;
  mesh.jacobian_matrix.irregular_element_number = mesh.irregular_element_number;
  mesh.jacobian_matrix.xix_regular = mesh.xix_regular;
  mesh.jacobian_matrix.jacobian_regular = mesh.jacobian_regular;

  // Reading the Jacobian matrix from the database file.
  try_read_array("read_xi_x", stream, mesh.jacobian_matrix.xix);
  try_read_array("read_xi_y", stream, mesh.jacobian_matrix.xiy);
  try_read_array("read_xi_z", stream, mesh.jacobian_matrix.xiz);
  try_read_array("read_eta_x", stream, mesh.jacobian_matrix.etax);
  try_read_array("read_eta_y", stream, mesh.jacobian_matrix.etay);
  try_read_array("read_eta_z", stream, mesh.jacobian_matrix.etaz);
  try_read_array("read_gamma_x", stream, mesh.jacobian_matrix.gammax);
  try_read_array("read_gamma_y", stream, mesh.jacobian_matrix.gammay);
  try_read_array("read_gamma_z", stream, mesh.jacobian_matrix.gammaz);
  try_read_array("read_jacobian", stream, mesh.jacobian_matrix.jacobian);

#ifndef NDEBUG
  // Print Jacobian matrix parameters and the first spectral element
  mpi->cout(mesh.jacobian_matrix.print());
  mpi->cout(mesh.jacobian_matrix.print(0, 0, 0, 0));

  // These print the Jacobian matrix array layout for the first spectral
  // element for debugging the array layout (Fortran v. C)
  // mpi->cout(mesh.jacobian_matrix.print(0, "xix"));
  // mpi->cout(mesh.jacobian_matrix.print(0, "etay"));
  // mpi->cout(mesh.jacobian_matrix.print(0, "gammaz"));
#endif

  // Marker that should be 10000
  check_read_test_value(stream, 10000);

  mesh.control_nodes =
      specfem::mesh::control_nodes<specfem::dimension::type::dim3>(
          mesh.parameters.nspec, mesh.parameters.ngnod, mesh.parameters.nnodes);

  // Read control nodes indexing
  try_read_control_nodes_indexing("read_control_nodes_indexing", stream,
                                  mesh.control_nodes.index_mapping);

  try_read_control_nodes_coordinates("read_control_nodes_coordinates", stream,
                                     mesh.control_nodes.coordinates);

#ifndef NDEBUG
  // Print control nodes parameters and the first spectral element
  mpi->cout(mesh.control_nodes.print());
#endif

  // Create material object
  mesh.materials = specfem::mesh::materials<specfem::dimension::type::dim3>(
      mesh.parameters.nspec, mesh.parameters.ngllx, mesh.parameters.nglly,
      mesh.parameters.ngllz, mesh.parameters.acoustic_simulation,
      mesh.parameters.elastic_simulation,
      mesh.parameters.poroelastic_simulation, mesh.parameters.anisotropy);

  try_read_array("read_kappa", stream, mesh.materials.kappa);
  try_read_array("read_mu", stream, mesh.materials.mu);

#ifndef NDEBUG
  // Print the materials
  mpi->cout(mesh.materials.print());
#endif

  int nacoustic, nelastic, nporoelastic;

  try_read_line("read_nacoustic", stream, &nacoustic);
  try_read_line("read_nelastic", stream, &nelastic);
  try_read_line("read_nporoelastic", stream, &nporoelastic);

  // Initialize element types object
  mesh.element_types =
      specfem::mesh::element_types<specfem::dimension::type::dim3>(
          mesh.parameters.nspec, nacoustic, nelastic, nporoelastic);

  // Reading the ispec type
  auto ispec_type = Kokkos::View<int *, Kokkos::HostSpace>("ispec_type", nspec);
  try_read_array("read_ispec_type", stream, ispec_type);

  // Compute the element arrays
  mesh.element_types.set_elements(ispec_type);

#ifndef NDEBUG
  // Print the element types
  mpi->cout(mesh.element_types.print());
  mpi->cout(mesh.element_types.print(0));

  // Print elements first element of each category
  mpi->cout(try_print_medium_element<specfem::element::medium_tag::acoustic>(
      mesh.element_types, 0));
  mpi->cout(try_print_medium_element<specfem::element::medium_tag::elastic>(
      mesh.element_types, 0));
  mpi->cout(try_print_medium_element<specfem::element::medium_tag::poroelastic>(
      mesh.element_types, 0));
#endif

  // Read test value 9999
  check_read_test_value(stream, 9999);

  // Intialize the mass matrix object
  mesh.mass_matrix = specfem::mesh::mass_matrix<specfem::dimension::type::dim3>(
      mesh.parameters.nglob, mesh.parameters.acoustic_simulation,
      mesh.parameters.elastic_simulation,
      mesh.parameters.poroelastic_simulation,
      mesh.parameters.approximate_ocean_load);

  // Read the acoustic mass matrix if acoustic simulation
  if (mesh.parameters.acoustic_simulation) {
    try_read_array("read_acoustic_mass_matrix", stream,
                   mesh.mass_matrix.acoustic);
  }

  // Read the density rho
  try_read_array("read_rho", stream, mesh.materials.rho);

  // Read test value 9998
  check_read_test_value(stream, 9998);

  // Read the elastic mass matrix if elastic simulation
  if (mesh.parameters.elastic_simulation) {
    try_read_array("read_elastic_mass_matrix", stream,
                   mesh.mass_matrix.elastic);

    // Read the stacey boundary values
    try_read_array("read_stacey_boundary_values", stream,
                   mesh.materials.rho_vp);
    try_read_array("read_stacey_boundary_values", stream,
                   mesh.materials.rho_vs);
  }

  // Read test value 9997
  check_read_test_value(stream, 9997);

  // If simulation poroelastic
  if (mesh.parameters.poroelastic_simulation) {
    try_read_array("read_poroelastic_mass_matrix_solid", stream,
                   mesh.mass_matrix.solid_poroelastic);
    try_read_array("read_poroelastic_mass_matrix_fluid", stream,
                   mesh.mass_matrix.fluid_poroelastic);

    // read the poroelastic material properties
    try_read_array("read_poroelastic_rho", stream, mesh.materials.poro_rho);
    try_read_array("read_poroelastic_kappa", stream, mesh.materials.poro_kappa);
    try_read_array("read_poroelastic_eta", stream, mesh.materials.poro_eta);
    try_read_array("read_poroelastic_tort", stream, mesh.materials.poro_tort);
    try_read_array("read_poroelastic_perm", stream, mesh.materials.poro_perm);
    try_read_array("read_poroelastic_phi", stream, mesh.materials.poro_phi);
    try_read_array("read_poroelastic_rho_vpI", stream,
                   mesh.materials.poro_rho_vpI);
    try_read_array("read_poroelastic_rho_vpII", stream,
                   mesh.materials.poro_rho_vpII);
    try_read_array("read_poroelastic_rho_vsI", stream,
                   mesh.materials.poro_rho_vsI);
  }

  // Read test value 9996
  check_read_test_value(stream, 9996);

  // Read number of absorbing boundaries
  int num_abs_boundary_faces;
  try_read_line("num_abs_boundary_faces", stream, &num_abs_boundary_faces);

  // Check whether the number of boundaries faces is equal to the number of
  // faces in the mesh parameters
  check_values("num_abs_boundary_faces", num_abs_boundary_faces,
               mesh.parameters.num_abs_boundary_faces);

  // if there are absorbing boundaries create the absorbing boundary object
  // and read the absorbing boundaries
  if (num_abs_boundary_faces > 0) {

    mesh.boundaries.absorbing_boundary =
        specfem::mesh::absorbing_boundary<specfem::dimension::type::dim3>(
            mesh.parameters.nglob, mesh.parameters.num_abs_boundary_faces,
            mesh.parameters.ngllsquare, mesh.parameters.acoustic_simulation,
            mesh.parameters.elastic_simulation, mesh.parameters.nspec2D_xmin,
            mesh.parameters.nspec2D_xmax, mesh.parameters.nspec2D_ymin,
            mesh.parameters.nspec2D_ymax, mesh.parameters.nspec2D_bottom,
            mesh.parameters.nspec2D_top);

    // Read the absorbing ispec
    try_read_index_array("abs_boundary_ispec", stream,
                         mesh.boundaries.absorbing_boundary.ispec);
    try_read_index_array("abs_boundary_ijk", stream,
                         mesh.boundaries.absorbing_boundary.ijk);
    try_read_array("abs_boundary_jacobian2Dw", stream,
                   mesh.boundaries.absorbing_boundary.jacobian2Dw);
    try_read_array("abs_boundary_normal", stream,
                   mesh.boundaries.absorbing_boundary.normal);

    // Read the absorbing mass matrix elastic

    if ((mesh.parameters.stacey_abc) & !(mesh.parameters.pml_abc)) {
      if (mesh.parameters.elastic_simulation) {
        try_read_array("abs_boundary_mass_elastic.mass", stream,
                       mesh.boundaries.absorbing_boundary.mass_elastic.x);
        try_read_array("abs_boundary_mass_elastic.mass", stream,
                       mesh.boundaries.absorbing_boundary.mass_elastic.y);
        try_read_array("abs_boundary_mass_elastic.mass", stream,
                       mesh.boundaries.absorbing_boundary.mass_elastic.z);
      }

      // Read the absorbing mass matrix acoustic
      if (mesh.parameters.acoustic_simulation) {
        try_read_array("abs_boundary_mass_acoustic.mass", stream,
                       mesh.boundaries.absorbing_boundary.mass_acoustic.mass);
      }
    }
#ifndef NDEBUG
    // Print the absorbing boundaries
    mpi->cout(mesh.boundaries.absorbing_boundary.print());

    // Print the absorbing boundaries for the first face
    // for debugging the array layout (Fortran v. C)
    // mpi->cout(mesh.boundaries.absorbing_boundary.print_ijk(0));
    // mpi->cout(mesh.boundaries.absorbing_boundary.print_ijk(num_abs_boundary_faces
    // - 1));
#endif
  }

  // Read test value 9995
  check_read_test_value(stream, 9995);

  // Read the number of boundaries
  int nspec2D_xmin, nspec2D_xmax, nspec2D_ymin, nspec2D_ymax, nspec2D_bottom,
      nspec2D_top;

  try_read_line("nspec2D_xmin", stream, &nspec2D_xmin);
  try_read_line("nspec2D_xmax", stream, &nspec2D_xmax);
  try_read_line("nspec2D_ymin", stream, &nspec2D_ymin);
  try_read_line("nspec2D_ymax", stream, &nspec2D_ymax);
  try_read_line("nspec2D_bottom", stream, &nspec2D_bottom);
  try_read_line("nspec2D_top", stream, &nspec2D_top);

#ifndef NDEBUG
  // Print the number of boundaries
  message = std::ostringstream();
  message << "nspec2D_xmin: " << nspec2D_xmin << "\n";
  message << "nspec2D_xmax: " << nspec2D_xmax << "\n";
  message << "nspec2D_ymin: " << nspec2D_ymin << "\n";
  message << "nspec2D_ymax: " << nspec2D_ymax << "\n";
  message << "nspec2D_bottom: " << nspec2D_bottom << "\n";
  message << "nspec2D_top: " << nspec2D_top << "\n";
  mpi->cout(message.str());
#endif

  // Check values
  check_values("nspec2D_xmin", nspec2D_xmin, mesh.parameters.nspec2D_xmin);
  check_values("nspec2D_xmax", nspec2D_xmax, mesh.parameters.nspec2D_xmax);
  check_values("nspec2D_ymin", nspec2D_ymin, mesh.parameters.nspec2D_ymin);
  check_values("nspec2D_ymax", nspec2D_ymax, mesh.parameters.nspec2D_ymax);
  check_values("nspec2D_bottom", nspec2D_bottom,
               mesh.parameters.nspec2D_bottom);
  check_values("nspec2D_top", nspec2D_top, mesh.parameters.nspec2D_top);

  if (mesh.parameters.num_abs_boundary_faces > 0) {

    if (nspec2D_xmin > 0) {
      try_read_index_array("ibelm_xmin", stream,
                           mesh.boundaries.absorbing_boundary.ibelm_xmin);
    }
    if (nspec2D_xmax > 0) {
      try_read_index_array("ibelm_xmax", stream,
                           mesh.boundaries.absorbing_boundary.ibelm_xmax);
    }
    if (nspec2D_ymin > 0) {
      try_read_index_array("ibelm_ymin", stream,
                           mesh.boundaries.absorbing_boundary.ibelm_ymin);
    }
    if (nspec2D_ymax > 0) {
      try_read_index_array("ibelm_ymax", stream,
                           mesh.boundaries.absorbing_boundary.ibelm_ymax);
    }
    if (nspec2D_bottom > 0) {
      try_read_index_array("ibelm_bottom", stream,
                           mesh.boundaries.absorbing_boundary.ibelm_bottom);
    }
    if (nspec2D_top > 0) {
      try_read_index_array("ibelm_top", stream,
                           mesh.boundaries.absorbing_boundary.ibelm_top);
    }
    // Print the absorbing boundaries
#ifndef NDEBUG
    mpi->cout(mesh.boundaries.absorbing_boundary.print());
#endif
  }

  // Read the number of free surface faces
  int num_free_surface_faces;
  try_read_line("num_free_surface_faces", stream, &num_free_surface_faces);
  check_values("num_free_surface_faces", num_free_surface_faces,
               mesh.parameters.num_free_surface_faces);

  // Create the free surface object
  mesh.boundaries.acoustic_free_surface =
      specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim3>(
          mesh.parameters.num_free_surface_faces, mesh.parameters.ngllsquare);

  // Read the free surface ispec
  try_read_index_array("acoustic_free_surface_ispec", stream,
                       mesh.boundaries.acoustic_free_surface.ispec);
  try_read_index_array("acoustic_free_surface_ijk", stream,
                       mesh.boundaries.acoustic_free_surface.ijk);
  try_read_array("acoustic_free_surface_jacobian2Dw", stream,
                 mesh.boundaries.acoustic_free_surface.jacobian2Dw);
  try_read_array("acoustic_free_surface_normal", stream,
                 mesh.boundaries.acoustic_free_surface.normal);

#ifndef NDEBUG
  // Print the free surface
  mpi->cout(mesh.boundaries.acoustic_free_surface.print());

  // Print the free surface for the first face
  // for debugging the array layout (Fortran v. C)
  // mpi->cout(mesh.acoustic_free_surface.print_ijk(0));
  // mpi->cout(mesh.acoustic_free_surface.print_ijk(num_free_surface_faces -
  // 1));
#endif

  // Create the coupled interfaces object
  mesh.coupled_interfaces =
      specfem::mesh::coupled_interfaces<specfem::dimension::type::dim3>(
          mesh.parameters.num_coupling_ac_el_faces,
          mesh.parameters.num_coupling_ac_po_faces,
          mesh.parameters.num_coupling_el_po_faces, mesh.parameters.ngllsquare);

  // Read the num_coupling_ac_el_faces
  int num_coupling_ac_el_faces;
  try_read_line("num_coupling_ac_el_faces", stream, &num_coupling_ac_el_faces);
  check_values("num_coupling_ac_el_faces", num_coupling_ac_el_faces,
               mesh.parameters.num_coupling_ac_el_faces);

  // Read the the coupling if the number is greater than 0
  if (mesh.coupled_interfaces.acoustic_elastic) {
    try_read_index_array(
        "coupling_ac_el_ispec", stream,
        mesh.coupled_interfaces.acoustic_elastic_interface.ispec);
    try_read_index_array(
        "coupling_ac_el_ijk", stream,
        mesh.coupled_interfaces.acoustic_elastic_interface.ijk);
    try_read_array(
        "coupling_ac_el_jacobian2Dw", stream,
        mesh.coupled_interfaces.acoustic_elastic_interface.jacobian2Dw);
    try_read_array("coupling_ac_el_normal", stream,
                   mesh.coupled_interfaces.acoustic_elastic_interface.normal);
  }

  // Read the num_coupling_ac_po_faces
  int num_coupling_ac_po_faces;
  try_read_line("num_coupling_ac_po_faces", stream, &num_coupling_ac_po_faces);

  // Check the values
  check_values("num_coupling_ac_po_faces", num_coupling_ac_po_faces,
               mesh.parameters.num_coupling_ac_po_faces);

  // Read the the coupling if the number is greater than 0
  if (mesh.coupled_interfaces.acoustic_poroelastic) {
    try_read_index_array(
        "coupling_ac_po_ispec", stream,
        mesh.coupled_interfaces.acoustic_poroelastic_interface.ispec);
    try_read_index_array(
        "coupling_ac_po_ijk", stream,
        mesh.coupled_interfaces.acoustic_poroelastic_interface.ijk);
    try_read_array(
        "coupling_ac_po_jacobian2Dw", stream,
        mesh.coupled_interfaces.acoustic_poroelastic_interface.jacobian2Dw);
    try_read_array(
        "coupling_ac_po_normal", stream,
        mesh.coupled_interfaces.acoustic_poroelastic_interface.normal);
  }

  // Read the num_coupling_el_po_faces
  int num_coupling_el_po_faces;
  try_read_line("num_coupling_el_po_faces", stream, &num_coupling_el_po_faces);

  // Check the values
  check_values("num_coupling_el_po_faces", num_coupling_el_po_faces,
               mesh.parameters.num_coupling_el_po_faces);

  // Read the the coupling if the number is greater than 0
  if (mesh.coupled_interfaces.elastic_poroelastic) {
    try_read_index_array(
        "coupling_el_po_ispec", stream,
        mesh.coupled_interfaces.elastic_poroelastic_interface.ispec);
    try_read_index_array(
        "coupling_el_po_ijk", stream,
        mesh.coupled_interfaces.elastic_poroelastic_interface.ijk);
    try_read_array(
        "coupling_el_po_jacobian2Dw", stream,
        mesh.coupled_interfaces.elastic_poroelastic_interface.jacobian2Dw);
    try_read_array(
        "coupling_el_po_normal", stream,
        mesh.coupled_interfaces.elastic_poroelastic_interface.normal);
    try_read_index_array(
        "coupling_po_el_ispec", stream,
        mesh.coupled_interfaces.poroelastic_elastic_interface.ispec);
    try_read_index_array(
        "coupling_po_el_ijk", stream,
        mesh.coupled_interfaces.poroelastic_elastic_interface.ijk);
    try_read_array(
        "coupling_po_el_jacobian2Dw", stream,
        mesh.coupled_interfaces.poroelastic_elastic_interface.jacobian2Dw);
    try_read_array(
        "coupling_po_el_normal", stream,
        mesh.coupled_interfaces.poroelastic_elastic_interface.normal);
  }

#ifndef NDEBUG
  // Print the interfaces
  mpi->cout(mesh.coupled_interfaces.print());
#endif

  // Read test value 9997
  check_read_test_value(stream, 9997);

  // Initialize MPI information
  int num_interfaces_ext_mesh, max_nibool_interfaces_ext_mesh;
  try_read_line("num_interfaces_ext_mesh", stream, &num_interfaces_ext_mesh);
  try_read_line("max_nibool_interfaces_ext_mesh", stream,
                &max_nibool_interfaces_ext_mesh);

  // Check values
  check_values("num_interfaces_ext_mesh", num_interfaces_ext_mesh,
               mesh.parameters.num_interfaces_ext_mesh);
  check_values("max_nibool_interfaces_ext_mesh", max_nibool_interfaces_ext_mesh,
               mesh.parameters.max_nibool_interfaces_ext_mesh);

  // If there are interfaces initialize the mpi object and read the interface
  // information
  if (mesh.parameters.num_interfaces_ext_mesh > 0) {
    mesh.mpi = specfem::mesh::mpi<specfem::dimension::type::dim3>(
        mesh.parameters.num_interfaces_ext_mesh,
        mesh.parameters.max_nibool_interfaces_ext_mesh);

    // Read the interface information
    try_read_index_array("mpi.neighbors", stream, mesh.mpi.neighbors);
    try_read_index_array("mpi.nibool", stream, mesh.mpi.nibool_interfaces);
    try_read_index_array("mpi.ibool", stream, mesh.mpi.ibool_interfaces);

  }
#ifndef NDEBUG
  else {
    mpi->cout("No MPI information stored in the binary file.\n");
  }
#endif

  // Read test value 9996
  check_read_test_value(stream, 9996);

  // Read the Anisotropic parameters if the simulation is anisotropic
  if (mesh.parameters.anisotropy) {
    try_read_array("read_anisotropic_parameter_c11", stream,
                   mesh.materials.c11);
    try_read_array("read_anisotropic_parameter_c12", stream,
                   mesh.materials.c12);
    try_read_array("read_anisotropic_parameter_c13", stream,
                   mesh.materials.c13);
    try_read_array("read_anisotropic_parameter_c14", stream,
                   mesh.materials.c14);
    try_read_array("read_anisotropic_parameter_c15", stream,
                   mesh.materials.c15);
    try_read_array("read_anisotropic_parameter_c16", stream,
                   mesh.materials.c16);
    try_read_array("read_anisotropic_parameter_c22", stream,
                   mesh.materials.c22);
    try_read_array("read_anisotropic_parameter_c23", stream,
                   mesh.materials.c23);
    try_read_array("read_anisotropic_parameter_c24", stream,
                   mesh.materials.c24);
    try_read_array("read_anisotropic_parameter_c25", stream,
                   mesh.materials.c25);
    try_read_array("read_anisotropic_parameter_c26", stream,
                   mesh.materials.c26);
    try_read_array("read_anisotropic_parameter_c33", stream,
                   mesh.materials.c33);
    try_read_array("read_anisotropic_parameter_c34", stream,
                   mesh.materials.c34);
    try_read_array("read_anisotropic_parameter_c35", stream,
                   mesh.materials.c35);
    try_read_array("read_anisotropic_parameter_c36", stream,
                   mesh.materials.c36);
    try_read_array("read_anisotropic_parameter_c44", stream,
                   mesh.materials.c44);
    try_read_array("read_anisotropic_parameter_c45", stream,
                   mesh.materials.c45);
    try_read_array("read_anisotropic_parameter_c46", stream,
                   mesh.materials.c46);
    try_read_array("read_anisotropic_parameter_c55", stream,
                   mesh.materials.c55);
    try_read_array("read_anisotropic_parameter_c56", stream,
                   mesh.materials.c56);
    try_read_array("read_anisotropic_parameter_c66", stream,
                   mesh.materials.c66);
  }

  // Check read test value 9995
  check_read_test_value(stream, 9995);

  // Initialize the inner outer object
  mesh.inner_outer = specfem::mesh::inner_outer<specfem::dimension::type::dim3>(
      mesh.parameters.nspec, mesh.parameters.acoustic_simulation,
      mesh.parameters.elastic_simulation,
      mesh.parameters.poroelastic_simulation);

  // Read ispec_is_inner
  try_read_array("ispec_is_inner", stream, mesh.inner_outer.ispec_is_inner);

  // Read inner outer acoustic elements
  if (mesh.parameters.acoustic_simulation) {
    int nspec_inner, nspec_outer, num_phase_ispec;

    try_read_line("nspec_inner_acoustic", stream, &nspec_inner, &nspec_outer);
    try_read_line("num_phase_ispec_acoustic", stream, &num_phase_ispec);

    // Initialize the inner outer elements
    mesh.inner_outer.inner_outer_acoustic = specfem::mesh::inner_outer_medium<
        specfem::element::medium_tag::acoustic>(nspec_inner, nspec_outer,
                                                num_phase_ispec);

    if (num_phase_ispec > 0) {
      try_read_index_array(
          "phase_ispec_inner_acoustic", stream,
          mesh.inner_outer.inner_outer_acoustic.phase_ispec_inner);
    }
  }

  // Check read test value 9994
  check_read_test_value(stream, 9994);

  // Read inner outer elastic elements
  if (mesh.parameters.elastic_simulation) {
    int nspec_inner, nspec_outer, num_phase_ispec;

    try_read_line("nspec_inner_elastic", stream, &nspec_inner, &nspec_outer);
    try_read_line("num_phase_ispec_elastic", stream, &num_phase_ispec);

    // Initialize the inner outer elements
    mesh.inner_outer.inner_outer_elastic = specfem::mesh::inner_outer_medium<
        specfem::element::medium_tag::elastic>(nspec_inner, nspec_outer,
                                               num_phase_ispec);

    if (num_phase_ispec > 0) {
      try_read_index_array(
          "phase_ispec_inner_elastic", stream,
          mesh.inner_outer.inner_outer_elastic.phase_ispec_inner);
    }
  }

  // Check read test value 9993
  check_read_test_value(stream, 9993);

  // Read inner outer poroelastic elements
  if (mesh.parameters.poroelastic_simulation) {
    int nspec_inner, nspec_outer, num_phase_ispec;

    try_read_line("nspec_inner_poroelastic", stream, &nspec_inner,
                  &nspec_outer);
    try_read_line("num_phase_ispec_poroelastic", stream, &num_phase_ispec);

    // Initialize the inner outer elements
    mesh.inner_outer.inner_outer_poroelastic =
        specfem::mesh::inner_outer_medium<
            specfem::element::medium_tag::poroelastic>(nspec_inner, nspec_outer,
                                                       num_phase_ispec);

    if (num_phase_ispec > 0) {
      try_read_index_array(
          "phase_ispec_inner_poroelastic", stream,
          mesh.inner_outer.inner_outer_poroelastic.phase_ispec_inner);
    }
  }

  // Check read test value 9992
  check_read_test_value(stream, 9992);

  // Read meshcoloring values if the mesh is colored
  if (mesh.parameters.use_mesh_coloring) {

    // initialize the mesh coloring object
    mesh.coloring = specfem::mesh::coloring<specfem::dimension::type::dim3>(
        mesh.parameters.acoustic_simulation,
        mesh.parameters.elastic_simulation);

    if (mesh.parameters.acoustic_simulation) {
      int num_colors_outer, num_colors_inner;
      try_read_line("num_colors_outer_acoustic", stream, &num_colors_outer);
      try_read_line("num_colors_inner_acoustic", stream, &num_colors_inner);

      // Initialize the medium coloring object
      mesh.coloring.coloring_acoustic = specfem::mesh::medium_coloring<
          specfem::element::medium_tag::acoustic>(num_colors_outer,
                                                  num_colors_inner);

      // Read the inner outer elements
      try_read_index_array("coloring_ispec_inner_acoustic", stream,
                           mesh.coloring.coloring_acoustic.elements);
    }

    if (mesh.parameters.acoustic_simulation) {
      int num_colors_outer, num_colors_inner;
      try_read_line("num_colors_outer_elastic", stream, &num_colors_outer);
      try_read_line("num_colors_inner_elastic", stream, &num_colors_inner);

      // Initialize the medium coloring object
      mesh.coloring.coloring_elastic =
          specfem::mesh::medium_coloring<specfem::element::medium_tag::elastic>(
              num_colors_outer, num_colors_inner);

      // Read the inner outer elements
      try_read_index_array("coloring_ispec_inner_elastic", stream,
                           mesh.coloring.coloring_elastic.elements);
    }
  }

  // Check read test value 9991
  check_read_test_value(stream, 9991);

  // Read the number of surface elements
  int nfaces_surface;
  try_read_line("nfaces_surface", stream, &nfaces_surface);

  // Initialize the surface elements object
  mesh.surface = specfem::mesh::surface<specfem::dimension::type::dim3>(
      nfaces_surface, mesh.parameters.nspec, mesh.parameters.nglob);

  // Read the surface elements
  try_read_array("ispec_is_surface", stream, mesh.surface.ispec_is_surface);
  try_read_array("iglob_is_surface", stream, mesh.surface.iglob_is_surface);

  // Check read test value 9990
  check_read_test_value(stream, 9990);

  // Read mesh adjacenecy
  int num_neighbors_all;
  try_read_line("num_neighbors_all", stream, &num_neighbors_all);

  // Initialize the adjacency object
  mesh.adjacency = specfem::mesh::adjacency<specfem::dimension::type::dim3>(
      mesh.parameters.nspec, num_neighbors_all);

  // Read the adjacency information
  try_read_index_array("neighbors_xadj", stream, mesh.adjacency.neighbors_xadj);
  try_read_index_array("neighbors_adjncy", stream,
                       mesh.adjacency.neighbors_adjncy);

  // Check read test value 9989
  check_read_test_value(stream, 9989);

  // Final print with basic information
  mpi->cout(mesh.print());

  stream.close();

  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim3>(
      mesh.element_types, mesh.boundaries, mesh.parameters);

  return mesh;
}
