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
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

void check_read_test_value(std::ifstream &stream, int test_value) {
  // Read test value that should be value
  int value;
  specfem::IO::fortran_read_line(stream, &value);
  if (test_value != value) {
    std::ostringstream error_message;
    error_message << "Test value (" << test_value << ") != read value ("
                  << value << "). "
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}

specfem::mesh::mesh<specfem::dimension::type::dim3>
specfem::IO::read_3d_mesh(const std::string mesh_parameters_file,
                          const std::string mesh_databases_file,
                          const specfem::MPI::MPI *mpi) {

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

  mesh.parameters.print();

  // Mesh Parameters are populated, closing the parameters file.
  stream.close();

  // Open the database file
  stream.open(mesh_databases_file);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open mesh database file");
  }

  int nspec, nglob, nspec_irregular;

  try {
    specfem::IO::fortran_read_line(stream, &nspec);
    specfem::IO::fortran_read_line(stream, &nglob);
    specfem::IO::fortran_read_line(stream, &nspec_irregular);

    std::tie(nspec, nglob, nspec_irregular) =
        std::make_tuple(nspec, nglob, nspec_irregular);

  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message
        << "Error reading nspec, nglob, nspec_irregular from database file: "
        << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
  }

  if (nspec != mesh.parameters.nspec) {
    std::ostringstream error_message;
    error_message << "Database nspec not equal to mesh parameters nspec. "
                  << "Database nspec = " << nspec
                  << " Mesh parameters nspec = " << mesh.parameters.nspec << "("
                  << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  if (nglob != mesh.parameters.nglob) {
    std::ostringstream error_message;
    error_message << "Database nglob not equal to mesh parameters nglob. "
                  << "Database nglob = " << nglob
                  << " Mesh parameters nglob = " << mesh.parameters.nglob << "("
                  << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  if (nspec_irregular != mesh.parameters.nspec_irregular) {
    std::ostringstream error_message;
    error_message << "Database nspec_irregular not equal to mesh parameters "
                     "nspec_irregular. "
                  << "Database nspec_irregular = " << nspec_irregular
                  << " Mesh parameters nspec_irregular = "
                  << mesh.parameters.nspec_irregular << "(" << __FILE__ << ":"
                  << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Create the mapping object
  mesh.mapping = specfem::mesh::mapping<specfem::dimension::type::dim3>(
      mesh.parameters.nspec, mesh.parameters.nglob,
      mesh.parameters.nspec_irregular, mesh.parameters.ngllx,
      mesh.parameters.nglly, mesh.parameters.ngllz);

  // Reading the mapping from the database file.
  try {
    specfem::IO::mesh::impl::fortran::dim3::read_index_array(
        stream, mesh.mapping.ibool);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading ibool from database file: " << e.what()
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Print Mapping parameters and the first spectral element
  mesh.mapping.print();
  mesh.mapping.print(0);
  mesh.mapping.print(mesh.parameters.nspec - 1);

  // Create the coordinates object
  mesh.coordinates = specfem::mesh::coordinates<specfem::dimension::type::dim3>(
      mesh.parameters.nspec, mesh.parameters.nglob, mesh.parameters.ngllx,
      mesh.parameters.nglly, mesh.parameters.ngllz);

  // Reading the coordinates from the database file.
  try {
    specfem::IO::mesh::impl::fortran::dim3::read_xyz(stream, mesh.coordinates,
                                                     mpi);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading xyz from database file: " << e.what() << "("
                  << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Print Coordinates parameters and the first global node
  mesh.coordinates.print();
  mesh.coordinates.print(0);
  mesh.coordinates.print(mesh.parameters.nglob - 1);

  // Initialize the partial derivatives object
  mesh.irregular_element_number = decltype(mesh.irregular_element_number)(
      "irregular_element_number", mesh.parameters.nspec_irregular);

  // Read Irregular elements
  try {
    specfem::IO::mesh::impl::fortran::dim3::read_index_array(
        stream, mesh.irregular_element_number);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message
        << "Error reading irregular_element_number from database file: "
        << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Read the partial derivatives (only two CUSTOM_REALs)
  try {
    fortran_read_line(stream, &mesh.xix_regular);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading xix_regular from database file: "
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Read the partial derivatives (only two CUSTOM_REALs)
  try {
    fortran_read_line(stream, &mesh.jacobian_regular);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading jacobian_regular from database file: "
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

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

  // Create the partial derivatives object
  mesh.partial_derivatives =
      specfem::mesh::partial_derivatives<specfem::dimension::type::dim3>(
          mesh.parameters.nspec, mesh.parameters.ngllx, mesh.parameters.nglly,
          mesh.parameters.ngllz);

  // Reading the partial derivatives from the database file.
  try {
    specfem::IO::mesh::impl::fortran::dim3::read_partial_derivatives(
        stream, mesh.partial_derivatives, mpi);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading partial derivatives from database file: "
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Print Partial Derivatives parameters and the first spectral element
  mesh.partial_derivatives.print();
  mesh.partial_derivatives.print(0, 0, 0, 0);

  // Marker that should be 10000
  try {
    check_read_test_value(stream, 10000);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading test value from database file: " << e.what()
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Create material object
  mesh.materials = specfem::mesh::materials<specfem::dimension::type::dim3>(
      mesh.parameters.nspec, mesh.parameters.ngllx, mesh.parameters.nglly,
      mesh.parameters.ngllz, mesh.parameters.acoustic_simulation,
      mesh.parameters.elastic_simulation,
      mesh.parameters.poroelastic_simulation, mesh.parameters.anisotropy);

  // Read Kappa using read_array
  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array(stream,
                                                       mesh.materials.kappa);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading kappa from database file: " << e.what()
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Read Mu using read_array
  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array(stream,
                                                       mesh.materials.mu);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading mu from database file: " << e.what() << "("
                  << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Print the materials
  mesh.materials.print();

  // Initialize element types object
  mesh.elements_types =
      specfem::mesh::element_types<specfem::dimension::type::dim3>(
          mesh.parameters.nspec);

  // Read boolean is x array!
  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array(
        stream, mesh.elements_types.ispec_is_acoustic);
    specfem::IO::mesh::impl::fortran::dim3::read_array(
        stream, mesh.elements_types.ispec_is_elastic);
    specfem::IO::mesh::impl::fortran::dim3::read_array(
        stream, mesh.elements_types.ispec_is_poroelastic);
    // This one allows for ispecs =
    // mesh.elements_types.get_elements<medium_tag>() to work
    mesh.elements_types.set_elements();
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading ispec_is_acoustic from database file: "
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Print the element types
  mesh.elements_types.print();
  mesh.elements_types.print(0);

  // Print elements first element of each category
  try {
    mesh.elements_types.print<specfem::element::medium_tag::acoustic>(0);
  } catch (std::runtime_error &e) {
    std::cout << e.what();
  };
  try {
    mesh.elements_types.print<specfem::element::medium_tag::elastic>(0);
  } catch (std::runtime_error &e) {
    std::cout << e.what();
  };
  try {
    mesh.elements_types.print<specfem::element::medium_tag::poroelastic>(0);
  } catch (std::runtime_error &e) {
    std::cout << e.what();
  };

  // read test value
  try {
    check_read_test_value(stream, 9999);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading test value from database file: " << e.what()
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Intialize the mass matrix object
  mesh.mass_matrix = specfem::mesh::mass_matrix<specfem::dimension::type::dim3>(
      mesh.parameters.nglob, mesh.parameters.acoustic_simulation,
      mesh.parameters.elastic_simulation,
      mesh.parameters.poroelastic_simulation,
      mesh.parameters.approximate_ocean_load);

  // Read the acoustic mass matrix if acoustic simulation
  if (mesh.parameters.acoustic_simulation) {
    try {
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.mass_matrix.acoustic);
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading acoustic mass matrix from database file: "
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }

  // Read the density rho
  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array(stream,
                                                       mesh.materials.rho);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading rho from database file: " << e.what() << "("
                  << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Read test value 9998
  try {
    check_read_test_value(stream, 9998);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading test value from database file: " << e.what()
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // Read the elastic mass matrix if elastic simulation
  if (mesh.parameters.elastic_simulation) {
    try {
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.mass_matrix.elastic);
      if (mesh.parameters.approximate_ocean_load) {
        specfem::IO::mesh::impl::fortran::dim3::read_array(
            stream, mesh.mass_matrix.ocean_load);
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading elastic mass matrix from database file: "
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }

    // Read the stacey boundary values
    try {
      specfem::IO::mesh::impl::fortran::dim3::read_array(stream,
                                                         mesh.materials.rho_vp);
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading rho_vp from database file: " << e.what()
                    << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }

    // Read the stacey boundary values
    try {
      specfem::IO::mesh::impl::fortran::dim3::read_array(stream,
                                                         mesh.materials.rho_vs);
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading rho_vs from database file: " << e.what()
                    << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }

  // Read test value 9997
  try {
    check_read_test_value(stream, 9997);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading test value from database file: " << e.what()
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  // If simulation poroelastic
  if (mesh.parameters.poroelastic_simulation) {
    try {
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.mass_matrix.solid_poroelastic);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.mass_matrix.fluid_poroelastic);
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message
          << "Error reading poroelastic mass matrix from database file: "
          << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }

    // read the poroelastic material properties
    try {
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_rho);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_kappa);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_eta);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_tort);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_perm);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_phi);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_rho_vpI);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_rho_vpII);
      specfem::IO::mesh::impl::fortran::dim3::read_array(
          stream, mesh.materials.poro_rho_vsI);
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading poroelastic material properties from "
                       "database file: "
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }

  // Read test value 9996
  try {
    check_read_test_value(stream, 9996);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading test value from database file: " << e.what()
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  stream.close();

  return mesh;
}
