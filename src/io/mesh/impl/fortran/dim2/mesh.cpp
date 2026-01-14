// Internal Includes
#include "specfem/mesh.hpp"
#include "enumerations/interface.hpp"
#include "io/fortranio/interface.hpp"
#include "io/interface.hpp"
#include "io/mesh/impl/fortran/dim2/read_adjacency_graph.hpp"
#include "io/mesh/impl/fortran/dim2/read_boundaries.hpp"
#include "io/mesh/impl/fortran/dim2/read_elements.hpp"
#include "io/mesh/impl/fortran/dim2/read_material_properties.hpp"
#include "io/mesh/impl/fortran/dim2/read_mesh_database.hpp"
#include "io/mesh/impl/fortran/dim2/read_parameters.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "specfem/logger.hpp"
#include "specfem_setup.hpp"

// External/Standard Libraries
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

specfem::mesh::mesh<specfem::dimension::type::dim2> specfem::io::read_2d_mesh(
    const std::string &filename,
    const specfem::enums::elastic_wave elastic_wave,
    const specfem::enums::electromagnetic_wave electromagnetic_wave) {

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
        specfem::io::mesh::impl::fortran::dim2::read_mesh_database_header(
            stream);
    mesh.nspec = nspec;
    mesh.npgeo = npgeo;
    mesh.nproc = nproc;
  } catch (std::runtime_error &e) {
    throw;
  }

  // Mesh class to be populated from the database file.
  try {
    mesh.control_nodes.coord =
        specfem::io::mesh::impl::fortran::dim2::read_coorg_elements(stream,
                                                                    mesh.npgeo);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.parameters =
        specfem::io::mesh::impl::fortran::dim2::read_mesh_parameters(stream);
  } catch (std::runtime_error &e) {
    throw;
  }

  mesh.control_nodes.ngnod = mesh.parameters.ngnod;
  mesh.control_nodes.nspec = mesh.nspec;
  mesh.control_nodes.knods = specfem::kokkos::HostView2d<int>(
      "specfem::mesh::knods", mesh.parameters.ngnod, mesh.nspec);

  int nspec_all = specfem::MPI::reduce(mesh.parameters.nspec, specfem::sum);
  int nelem_acforcing_all =
      specfem::MPI::reduce(mesh.parameters.nelem_acforcing, specfem::sum);
  int nelem_acoustic_surface_all = specfem::MPI::reduce(
      mesh.parameters.nelem_acoustic_surface, specfem::sum);

  try {
    auto [n_sls, attenuation_f0_reference, read_velocities_at_f0] =
        specfem::io::mesh::impl::fortran::dim2::read_mesh_database_attenuation(
            stream);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.materials =
        specfem::io::mesh::impl::fortran::dim2::read_material_properties(
            stream, mesh.parameters.numat, mesh.nspec, elastic_wave,
            electromagnetic_wave, mesh.control_nodes.knods);
  } catch (std::runtime_error &e) {
    throw;
  }

  int ninterfaces;
  int max_interface_size;

  specfem::io::fortran_read_line(stream, &ninterfaces, &max_interface_size);

  try {
    mesh.boundaries = specfem::io::mesh::impl::fortran::dim2::read_boundaries(
        stream, mesh.parameters.nspec, mesh.parameters.nelemabs,
        mesh.parameters.nelem_acoustic_surface, mesh.parameters.nelem_acforcing,
        mesh.control_nodes.knods);
  } catch (std::runtime_error &e) {
    throw;
  }

  std::set<std::pair<int, int> > coupled_interfaces;
  try {
    for (const auto &num_interfaces :
         { mesh.parameters.num_fluid_solid_edges,
           mesh.parameters.num_fluid_poro_edges,
           mesh.parameters.num_solid_poro_edges }) {
      int medium1_ispec_l, medium2_ispec_l;
      for (int i = 0; i < num_interfaces; i++) {
        specfem::io::fortran_read_line(stream, &medium2_ispec_l,
                                       &medium1_ispec_l);
        coupled_interfaces.insert(
            std::make_pair(medium1_ispec_l - 1, medium2_ispec_l - 1));
      };
    }
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.tangential_nodes =
        specfem::io::mesh::impl::fortran::dim2::read_tangential_elements(
            stream, mesh.parameters.nnodes_tangential_curve);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.axial_nodes =
        specfem::io::mesh::impl::fortran::dim2::read_axial_elements(
            stream, mesh.parameters.nelem_on_the_axis, mesh.nspec);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.adjacency_graph =
        specfem::io::mesh::impl::fortran::dim2::read_adjacency_graph(mesh.nspec,
                                                                     stream);
  } catch (std::runtime_error &e) {
    throw;
  }

  // Check if database file was read completely
  if (stream.get() && !stream.eof()) {
    throw std::runtime_error("The Database file wasn't fully read. Is there "
                             "anything written after axial elements?");
  }

  stream.close();

  // Setup and verify coupled interfaces
  mesh.setup_coupled_interfaces(coupled_interfaces);

  // Print material properties

  specfem::Logger::debug("Material systems:\n"
                         "------------------------------");

  specfem::Logger::debug("Number of material systems = " +
                         std::to_string(mesh.materials.n_materials) + "\n\n");

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T,
                  ELECTROMAGNETIC_TE),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
        for (const auto material :
             mesh.materials.get_container<_medium_tag_, _property_tag_>()
                 .element_materials) {
          specfem::Logger::debug(material.print());
        }
      })

  int total_materials_read = 0;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T,
                  ELECTROMAGNETIC_TE),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
        total_materials_read +=
            mesh.materials.get_container<_medium_tag_, _property_tag_>()
                .element_materials.size();
      })

  if (total_materials_read != mesh.materials.n_materials) {
    std::ostringstream message;
    message << "Total number of materials read = " << total_materials_read
            << "\n"
            << "Total number of materials in the database = "
            << mesh.materials.n_materials << "\n";

    throw std::runtime_error(message.str());
  }

  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
      mesh.materials, mesh.boundaries);

  mesh.check_consistency();

  return mesh;
}
