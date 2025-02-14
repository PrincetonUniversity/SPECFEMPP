// Internal Includes
#include "mesh/mesh.hpp"
#include "IO/fortranio/interface.hpp"
#include "IO/interface.hpp"
#include "IO/mesh/impl/fortran/dim2/read_boundaries.hpp"
#include "IO/mesh/impl/fortran/dim2/read_elements.hpp"
#include "IO/mesh/impl/fortran/dim2/read_interfaces.hpp"
#include "IO/mesh/impl/fortran/dim2/read_material_properties.hpp"
#include "IO/mesh/impl/fortran/dim2/read_mesh_database.hpp"
#include "IO/mesh/impl/fortran/dim2/read_parameters.hpp"
#include "IO/mesh/impl/fortran/dim3/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "mesh/tags/tags.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"

// External/Standard Libraries
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

specfem::mesh::mesh<specfem::dimension::type::dim2>
specfem::IO::read_2d_mesh(const std::string filename,
                          const specfem::MPI::MPI *mpi) {

  // Declaring empty mesh objects
  specfem::mesh::mesh<specfem::dimension::type::dim2> mesh;

  // Open the database file
  std::ifstream stream;
  stream.open(filename);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open database file");
  }
  int nspec, npgeo, nproc;

  try {
    std::tie(nspec, npgeo, nproc) =
        specfem::IO::mesh::impl::fortran::dim2::read_mesh_database_header(
            stream, mpi);
    mesh.nspec = nspec;
    mesh.npgeo = npgeo;
    mesh.nproc = nproc;
  } catch (std::runtime_error &e) {
    throw;
  }

  // Mesh class to be populated from the database file.
  try {
    mesh.control_nodes.coord =
        specfem::IO::mesh::impl::fortran::dim2::read_coorg_elements(
            stream, mesh.npgeo, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.parameters =
        specfem::IO::mesh::impl::fortran::dim2::read_mesh_parameters(stream,
                                                                     mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mesh.control_nodes.ngnod = mesh.parameters.ngnod;
  mesh.control_nodes.nspec = mesh.nspec;
  mesh.control_nodes.knods = specfem::kokkos::HostView2d<int>(
      "specfem::mesh::knods", mesh.parameters.ngnod, mesh.nspec);

  int nspec_all = mpi->reduce(mesh.parameters.nspec, specfem::MPI::sum);
  int nelem_acforcing_all =
      mpi->reduce(mesh.parameters.nelem_acforcing, specfem::MPI::sum);
  int nelem_acoustic_surface_all =
      mpi->reduce(mesh.parameters.nelem_acoustic_surface, specfem::MPI::sum);

  try {
    auto [n_sls, attenuation_f0_reference, read_velocities_at_f0] =
        specfem::IO::mesh::impl::fortran::dim2::read_mesh_database_attenuation(
            stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.materials =
        specfem::IO::mesh::impl::fortran::dim2::read_material_properties(
            stream, mesh.parameters.numat, mesh.nspec, mesh.control_nodes.knods,
            mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // try {
  //   materials = specfem::mesh::IO::fortran::read_material_properties(
  //       stream, this->parameters.numat, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   this->material_ind = specfem::mesh::material_ind(
  //       stream, this->parameters.ngnod, this->nspec, this->parameters.numat,
  //       this->control_nodes.knods, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   this->interface = specfem::mesh::interfaces::interface(stream, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  int ninterfaces;
  int max_interface_size;

  specfem::IO::fortran_read_line(stream, &ninterfaces, &max_interface_size);

  try {
    mesh.boundaries = specfem::IO::mesh::impl::fortran::dim2::read_boundaries(
        stream, mesh.parameters.nspec, mesh.parameters.nelemabs,
        mesh.parameters.nelem_acoustic_surface, mesh.parameters.nelem_acforcing,
        mesh.control_nodes.knods, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // try {
  //   this->boundaries.absorbing_boundary = specfem::mesh::absorbing_boundary(
  //       stream, this->parameters.nelemabs, this->parameters.nspec, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   this->boundaries.forcing_boundary = specfem::mesh::forcing_boundary(
  //       stream, this->parameters.nelem_acforcing, this->parameters.nspec,
  //       mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   this->boundaries.acoustic_free_surface =
  //       specfem::mesh::acoustic_free_surface(
  //           stream, this->parameters.nelem_acoustic_surface,
  //           this->control_nodes.knods, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  try {
    mesh.coupled_interfaces =
        specfem::IO::mesh::impl::fortran::dim2::read_coupled_interfaces(
            stream, mesh.parameters.num_fluid_solid_edges,
            mesh.parameters.num_fluid_poro_edges,
            mesh.parameters.num_solid_poro_edges, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.tangential_nodes =
        specfem::IO::mesh::impl::fortran::dim2::read_tangential_elements(
            stream, mesh.parameters.nnodes_tangential_curve);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.axial_nodes =
        specfem::IO::mesh::impl::fortran::dim2::read_axial_elements(
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

  // Print material properties

  mpi->cout("Material systems:\n"
            "------------------------------");

  mpi->cout("Number of material systems = " +
            std::to_string(mesh.materials.n_materials) + "\n\n");

  const auto l_elastic_isotropic =
      mesh.materials.elastic_isotropic.material_properties;
  const auto l_acoustic_isotropic =
      mesh.materials.acoustic_isotropic.material_properties;

  const auto l_elastic_anisotropic =
      mesh.materials.elastic_anisotropic.material_properties;

  for (const auto material : l_elastic_isotropic) {
    mpi->cout(material.print());
  }

  for (const auto material : l_acoustic_isotropic) {
    mpi->cout(material.print());
  }

  for (const auto material : l_elastic_anisotropic) {
    mpi->cout(material.print());
  }

  assert(l_elastic_isotropic.size() + l_acoustic_isotropic.size() +
             l_elastic_anisotropic.size() ==
         mesh.materials.n_materials);

  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
      mesh.materials, mesh.boundaries);

  return mesh;
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
    specfem::IO::mesh::impl::fortran::dim3::read_index_array(stream,
                                                             mapping.ibool);
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
        stream, mapping.irregular_element_number);
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

  stream.close();

  // // Mesh class to be populated from the database file.
  // try {
  //   mesh.control_nodes.coord =
  //       specfem::IO::mesh::impl::fortran::dim2::read_coorg_elements(stream,
  //                                                             mesh.npgeo,
  //                                                             mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   mesh.parameters =
  //       specfem::IO::mesh::impl::fortran::dim2::read_mesh_parameters(stream,
  //       mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // mesh.control_nodes.ngnod = mesh.parameters.ngnod;
  // mesh.control_nodes.nspec = mesh.nspec;
  // mesh.control_nodes.knods = specfem::kokkos::HostView2d<int>(
  //     "specfem::mesh::knods", mesh.parameters.ngnod, mesh.nspec);

  // int nspec_all = mpi->reduce(mesh.parameters.nspec, specfem::MPI::sum);
  // int nelem_acforcing_all =
  //     mpi->reduce(mesh.parameters.nelem_acforcing, specfem::MPI::sum);
  // int nelem_acoustic_surface_all =
  //     mpi->reduce(mesh.parameters.nelem_acoustic_surface, specfem::MPI::sum);

  // try {
  //   auto [n_sls, attenuation_f0_reference, read_velocities_at_f0] =
  //       specfem::IO::mesh::impl::fortran::dim2::read_mesh_database_attenuation(stream,
  //                                                                        mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   mesh.materials =
  //   specfem::IO::mesh::impl::fortran::dim2::read_material_properties(
  //       stream, mesh.parameters.numat, mesh.nspec, mesh.control_nodes.knods,
  //       mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // // try {
  // //   materials = specfem::mesh::IO::fortran::read_material_properties(
  // //       stream, this->parameters.numat, mpi);
  // // } catch (std::runtime_error &e) {
  // //   throw;
  // // }

  // // try {
  // //   this->material_ind = specfem::mesh::material_ind(
  // //       stream, this->parameters.ngnod, this->nspec,
  // this->parameters.numat,
  // //       this->control_nodes.knods, mpi);
  // // } catch (std::runtime_error &e) {
  // //   throw;
  // // }

  // // try {
  // //   this->interface = specfem::mesh::interfaces::interface(stream, mpi);
  // // } catch (std::runtime_error &e) {
  // //   throw;
  // // }

  // int ninterfaces;
  // int max_interface_size;

  // specfem::IO::fortran_read_line(stream, &ninterfaces, &max_interface_size);

  // try {
  //   mesh.boundaries =
  //   specfem::IO::mesh::impl::fortran::dim2::read_boundaries(
  //       stream, mesh.parameters.nspec, mesh.parameters.nelemabs,
  //       mesh.parameters.nelem_acoustic_surface,
  //       mesh.parameters.nelem_acforcing, mesh.control_nodes.knods, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // // try {
  // //   this->boundaries.absorbing_boundary =
  // specfem::mesh::absorbing_boundary(
  // //       stream, this->parameters.nelemabs, this->parameters.nspec, mpi);
  // // } catch (std::runtime_error &e) {
  // //   throw;
  // // }

  // // try {
  // //   this->boundaries.forcing_boundary = specfem::mesh::forcing_boundary(
  // //       stream, this->parameters.nelem_acforcing, this->parameters.nspec,
  // //       mpi);
  // // } catch (std::runtime_error &e) {
  // //   throw;
  // // }

  // // try {
  // //   this->boundaries.acoustic_free_surface =
  // //       specfem::mesh::acoustic_free_surface(
  // //           stream, this->parameters.nelem_acoustic_surface,
  // //           this->control_nodes.knods, mpi);
  // // } catch (std::runtime_error &e) {
  // //   throw;
  // // }

  // try {
  //   mesh.coupled_interfaces =
  //       specfem::IO::mesh::impl::fortran::dim2::read_coupled_interfaces(
  //           stream, mesh.parameters.num_fluid_solid_edges,
  //           mesh.parameters.num_fluid_poro_edges,
  //           mesh.parameters.num_solid_poro_edges, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   mesh.tangential_nodes =
  //       specfem::IO::mesh::impl::fortran::dim2::read_tangential_elements(
  //           stream, mesh.parameters.nnodes_tangential_curve);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   mesh.axial_nodes =
  //   specfem::IO::mesh::impl::fortran::dim2::read_axial_elements(
  //       stream, mesh.parameters.nelem_on_the_axis, mesh.nspec, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // // Check if database file was read completely
  // if (stream.get() && !stream.eof()) {
  //   throw std::runtime_error("The Database file wasn't fully read. Is there "
  //                            "anything written after axial elements?");
  // }

  // stream.close();

  // // Print material properties

  // mpi->cout("Material systems:\n"
  //           "------------------------------");

  // mpi->cout("Number of material systems = " +
  //           std::to_string(mesh.materials.n_materials) + "\n\n");

  // const auto l_elastic_isotropic =
  //     mesh.materials.elastic_isotropic.material_properties;
  // const auto l_acoustic_isotropic =
  //     mesh.materials.acoustic_isotropic.material_properties;

  // const auto l_elastic_anisotropic =
  //     mesh.materials.elastic_anisotropic.material_properties;

  // for (const auto material : l_elastic_isotropic) {
  //   mpi->cout(material.print());
  // }

  // for (const auto material : l_acoustic_isotropic) {
  //   mpi->cout(material.print());
  // }

  // for (const auto material : l_elastic_anisotropic) {
  //   mpi->cout(material.print());
  // }

  // assert(l_elastic_isotropic.size() + l_acoustic_isotropic.size() +
  //            l_elastic_anisotropic.size() ==
  //        mesh.materials.n_materials);

  // mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
  //     mesh.materials, mesh.boundaries);

  return mesh;
}
