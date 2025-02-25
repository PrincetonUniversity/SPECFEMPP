// Internal Includes
#include "mesh/mesh.hpp"
#include "IO/fortranio/interface.hpp"
#include "IO/interface.hpp"
#include "IO/mesh/impl/fortran/dim3/interface.hpp"
#include "enumerations/interface.hpp"
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
void try_print_medium_element(
    const specfem::mesh::element_types<specfem::dimension::type::dim3>
        &elements_types,
    int index) {
  try {
    elements_types.print<medium>(index);
  } catch (std::runtime_error &e) {
    std::cout << e.what();
  } catch (...) {
    std::cout << "Unknown exception caught in try_print_medium_element"
              << std::endl;
  }
}

specfem::mesh::mesh<specfem::dimension::type::dim3>
specfem::IO::read_3d_mesh(const std::string mesh_parameters_file,
                          const std::string mesh_databases_file,
                          const specfem::MPI::MPI *mpi) {

  // Creating aliases for Checking functions
  using specfem::IO::mesh::impl::fortran::dim3::check_read_test_value;
  using specfem::IO::mesh::impl::fortran::dim3::check_values;

  // Creating aliases for Array Reading functions
  using specfem::IO::mesh::impl::fortran::dim3::try_read_array;
  using specfem::IO::mesh::impl::fortran::dim3::try_read_index_array;
  using specfem::IO::mesh::impl::fortran::dim3::try_read_line;

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
        specfem::IO::mesh::impl::fortran::dim3::read_mesh_parameters(stream,
                                                                     mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // Mesh Parameters are populated, closing the parameters file.
  stream.close();

#ifndef NDEBUG
  // Print the parameters
  mesh.parameters.print();
#endif

  // Open the database file
  stream.open(mesh_databases_file);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open mesh database file");
  }

  int nspec, nglob, nspec_irregular;

  try_read_line("read_nspec", stream, &nspec);
  try_read_line("read_nglob", stream, &nglob);
  try_read_line("read_nspec_irregular", stream, &nspec_irregular);

  // Check values
  check_values("nspec", nspec, mesh.parameters.nspec);
  check_values("nglob", nglob, mesh.parameters.nglob);
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
  mesh.mapping.print();
  mesh.mapping.print(0);
  mesh.mapping.print(mesh.parameters.nspec - 1);
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
  mesh.coordinates.print();
  mesh.coordinates.print(0);
  mesh.coordinates.print(mesh.parameters.nglob - 1);
#endif

  // Initialize the partial derivatives object
  mesh.irregular_element_number = decltype(mesh.irregular_element_number)(
      "irregular_element_number", mesh.parameters.nspec_irregular);

  // Read Irregular elements
  try_read_index_array("read_irregular_element_number", stream,
                       mesh.irregular_element_number);

  // Read the partial derivatives (only two CUSTOM_REALs)
  try_read_line("read_xix_regular", stream, &mesh.xix_regular);
  try_read_line("read_jacobian_regular", stream, &mesh.jacobian_regular);

#ifndef NDEBUG
  // Print the first and last irregular element
  std::cout << "First irregular element: " << mesh.irregular_element_number(0)
            << std::endl;
  std::cout << "Last irregular element: "
            << mesh.irregular_element_number(mesh.parameters.nspec_irregular -
                                             1)
            << std::endl;

  // Print xix and jacobian
  std::cout << "xix_regular: " << mesh.xix_regular << std::endl;
  std::cout << "jacobian_regular: " << mesh.jacobian_regular << std::endl;
#endif

  // Create the partial derivatives object
  mesh.partial_derivatives =
      specfem::mesh::partial_derivatives<specfem::dimension::type::dim3>(
          mesh.parameters.nspec, mesh.parameters.ngllx, mesh.parameters.nglly,
          mesh.parameters.ngllz);

  // Reading the partial derivatives from the database file.
  try_read_array("read_xi_x", stream, mesh.partial_derivatives.xix);
  try_read_array("read_xi_y", stream, mesh.partial_derivatives.xiy);
  try_read_array("read_xi_z", stream, mesh.partial_derivatives.xiz);
  try_read_array("read_eta_x", stream, mesh.partial_derivatives.etax);
  try_read_array("read_eta_y", stream, mesh.partial_derivatives.etay);
  try_read_array("read_eta_z", stream, mesh.partial_derivatives.etaz);
  try_read_array("read_gamma_x", stream, mesh.partial_derivatives.gammax);
  try_read_array("read_gamma_y", stream, mesh.partial_derivatives.gammay);
  try_read_array("read_gamma_z", stream, mesh.partial_derivatives.gammaz);
  try_read_array("read_jacobian", stream, mesh.partial_derivatives.jacobian);

#ifndef NDEBUG
  // Print Partial Derivatives parameters and the first spectral element
  mesh.partial_derivatives.print();
  mesh.partial_derivatives.print(0, 0, 0, 0);
#endif

  // Marker that should be 10000
  check_read_test_value(stream, 10000);

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
  mesh.materials.print();
#endif

  // Initialize element types object
  mesh.elements_types =
      specfem::mesh::element_types<specfem::dimension::type::dim3>(
          mesh.parameters.nspec);

  // Read boolean is x array!
  try_read_array("read_ispec_is_acoustic", stream,
                 mesh.elements_types.ispec_is_acoustic);
  try_read_array("read_ispec_is_elastic", stream,
                 mesh.elements_types.ispec_is_elastic);
  try_read_array("read_ispec_is_poroelastic", stream,
                 mesh.elements_types.ispec_is_poroelastic);

  // Compute the element arrays
  mesh.elements_types.set_elements();

#ifndef NDEBUG
  // Print the element types
  mesh.elements_types.print();
  mesh.elements_types.print(0);

  // Print elements first element of each category
  try_print_medium_element<specfem::element::medium_tag::acoustic>(
      mesh.elements_types, 0);
  try_print_medium_element<specfem::element::medium_tag::elastic>(
      mesh.elements_types, 0);
  try_print_medium_element<specfem::element::medium_tag::poroelastic>(
      mesh.elements_types, 0);
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
    mesh.absorbing_boundary =
        specfem::mesh::absorbing_boundary<specfem::dimension::type::dim3>(
            mesh.parameters.nglob, mesh.parameters.num_abs_boundary_faces,
            mesh.parameters.ngllsquare, mesh.parameters.acoustic_simulation,
            mesh.parameters.elastic_simulation, mesh.parameters.nspec2D_xmin,
            mesh.parameters.nspec2D_xmax, mesh.parameters.nspec2D_ymin,
            mesh.parameters.nspec2D_ymax, mesh.parameters.nspec2D_bottom,
            mesh.parameters.nspec2D_top);

    // Read the absorbing ispec
    try_read_index_array("abs_boundary_ispec", stream,
                         mesh.absorbing_boundary.ispec);
    try_read_index_array("abs_bounary_ijk", stream,
                         mesh.absorbing_boundary.ijk);
    try_read_array("abs_boundary_jacobian2Dw", stream,
                   mesh.absorbing_boundary.jacobian2Dw);
    try_read_array("abs_boundary_normal", stream,
                   mesh.absorbing_boundary.normal);

    // Read the absorbing mass matrix elastic
    if (mesh.parameters.elastic_simulation) {
      try_read_array("abs_boundary_mass_elastic.mass", stream,
                     mesh.absorbing_boundary.mass_elastic.x);
      try_read_array("abs_boundary_mass_elastic.mass", stream,
                     mesh.absorbing_boundary.mass_elastic.y);
      try_read_array("abs_boundary_mass_elastic.mass", stream,
                     mesh.absorbing_boundary.mass_elastic.z);
    }

    // Read the absorbing mass matrix acoustic
    if (mesh.parameters.acoustic_simulation) {
      try_read_array("abs_boundary_mass_acoustic.mass", stream,
                     mesh.absorbing_boundary.mass_acoustic.mass);
    }
#ifndef NDEBUG
    // Print the absorbing boundaries
    // mesh.absorbing_boundary.print();
    // mesh.absorbing_boundary.print(0);
    // mesh.absorbing_boundary.print(num_abs_boundary_faces - 1);
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
  std::cout << "nspec2D_xmin: " << nspec2D_xmin << std::endl;
  std::cout << "nspec2D_xmax: " << nspec2D_xmax << std::endl;
  std::cout << "nspec2D_ymin: " << nspec2D_ymin << std::endl;
  std::cout << "nspec2D_ymax: " << nspec2D_ymax << std::endl;
  std::cout << "nspec2D_bottom: " << nspec2D_bottom << std::endl;
  std::cout << "nspec2D_top: " << nspec2D_top << std::endl;
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
                           mesh.absorbing_boundary.ibelm_xmin);
    }
    if (nspec2D_xmax > 0) {
      try_read_index_array("ibelm_xmax", stream,
                           mesh.absorbing_boundary.ibelm_xmax);
    }
    if (nspec2D_ymin > 0) {
      try_read_index_array("ibelm_ymin", stream,
                           mesh.absorbing_boundary.ibelm_ymin);
    }
    if (nspec2D_ymax > 0) {
      try_read_index_array("ibelm_ymax", stream,
                           mesh.absorbing_boundary.ibelm_ymax);
    }
    if (nspec2D_bottom > 0) {
      try_read_index_array("ibelm_bottom", stream,
                           mesh.absorbing_boundary.ibelm_bottom);
    }
    if (nspec2D_top > 0) {
      try_read_index_array("ibelm_top", stream,
                           mesh.absorbing_boundary.ibelm_top);
    }
    // Print the abosrbing boundaries
#ifndef NDEBUG
    mesh.absorbing_boundary.print();
#endif
  }

  // Read the number of free surface faces
  int num_free_surface_faces;
  try_read_line("num_free_surface_faces", stream, &num_free_surface_faces);
  check_values("num_free_surface_faces", num_free_surface_faces,
               mesh.parameters.num_free_surface_faces);

  // Create the free surface object
  mesh.free_surface =
      specfem::mesh::free_surface<specfem::dimension::type::dim3>(
          mesh.parameters.num_free_surface_faces, mesh.parameters.ngllsquare);

  // Read the free surface ispec
  try_read_index_array("free_surface_ispec", stream, mesh.free_surface.ispec);
  try_read_index_array("free_surface_ijk", stream, mesh.free_surface.ijk);
  try_read_array("free_surface_jacobian2Dw", stream,
                 mesh.free_surface.jacobian2Dw);
  try_read_array("free_surface_normal", stream, mesh.free_surface.normal);

#ifndef NDEBUG
  // Print the free surface
  mesh.free_surface.print();
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
  mesh.coupled_interfaces.print();
#endif

  // Read test value 9997
  check_read_test_value(stream, 9997);

  stream.close();

  return mesh;
}
