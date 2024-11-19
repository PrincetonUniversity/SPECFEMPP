#include "IO/mesh/read_mesh.hpp"
#include "IO/fortranio/interface.hpp"
#include "IO/mesh/fortran/read_boundaries.hpp"
#include "IO/mesh/fortran/read_elements.hpp"
#include "IO/mesh/fortran/read_interfaces.hpp"
#include "IO/mesh/fortran/read_material_properties.hpp"
#include "IO/mesh/fortran/read_mesh_database.hpp"
#include "IO/mesh/fortran/read_properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "material/material.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

specfem::mesh::mesh specfem::IO::read_mesh(const std::string filename,
                                           const specfem::MPI::MPI *mpi) {

  std::ifstream stream;
  stream.open(filename);
  int nspec, npgeo, nproc;

  specfem::mesh::control_nodes control_nodes;
  specfem::mesh::properties parameters;
  specfem::mesh::materials materials;
  specfem::mesh::boundaries boundaries;
  specfem::mesh::elements::axial_elements axial_nodes;
  specfem::mesh::elements::tangential_elements tangential_nodes;
  specfem::mesh::coupled_interfaces coupled_interfaces;

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open database file");
  }

  try {
    auto [nspec, npgeo, nproc] =
        specfem::IO::mesh::fortran::read_mesh_database_header(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // Mesh class to be populated from the database file.
  try {

    control_nodes.coord =
        specfem::IO::mesh::fortran::read_coorg_elements(stream, npgeo, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    auto parameters = specfem::IO::mesh::fortran::read_properties(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  control_nodes.ngnod = parameters.ngnod;
  control_nodes.nspec = nspec;
  control_nodes.knods = specfem::kokkos::HostView2d<int>(
      "specfem::mesh::knods", parameters.ngnod, nspec);

  int nspec_all = mpi->reduce(parameters.nspec, specfem::MPI::sum);
  int nelem_acforcing_all =
      mpi->reduce(parameters.nelem_acforcing, specfem::MPI::sum);
  int nelem_acoustic_surface_all =
      mpi->reduce(parameters.nelem_acoustic_surface, specfem::MPI::sum);

  try {
    auto [n_sls, attenuation_f0_reference, read_velocities_at_f0] =
        specfem::IO::mesh::fortran::read_mesh_database_attenuation(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    auto materials = specfem::IO::mesh::fortran::read_material_properties(
        stream, nspec, parameters.numat, control_nodes.knods, mpi);
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
    auto boundaries = specfem::IO::mesh::fortran::read_boundaries(
        stream, parameters.nspec, parameters.nelemabs,
        parameters.nelem_acoustic_surface, parameters.nelem_acforcing,
        control_nodes.knods, mpi);
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
    auto coupled_interfaces =
        specfem::IO::mesh::fortran::read_coupled_interfaces(
            stream, parameters.num_fluid_solid_edges,
            parameters.num_fluid_poro_edges, parameters.num_solid_poro_edges,
            mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    auto tangential_nodes =
        specfem::IO::mesh::fortran::read_tangential_elements(
            stream, parameters.nnodes_tangential_curve);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    auto axial_nodes = specfem::IO::mesh::fortran::read_axial_elements(
        stream, parameters.nelem_on_the_axis, nspec, mpi);
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
            std::to_string(materials.n_materials) + "\n\n");

  const auto l_elastic_isotropic =
      materials.elastic_isotropic.material_properties;
  const auto l_acoustic_isotropic =
      materials.acoustic_isotropic.material_properties;

  for (const auto material : l_elastic_isotropic) {
    mpi->cout(material.print());
  }

  for (const auto material : l_acoustic_isotropic) {
    mpi->cout(material.print());
  }

  assert(l_elastic_isotropic.size() + l_acoustic_isotropic.size() ==
         materials.n_materials);

  auto tags = specfem::mesh::tags(materials, boundaries);

  return specfem::mesh::mesh(npgeo, nspec, nproc, control_nodes, parameters,
                             coupled_interfaces, boundaries, tags,
                             tangential_nodes, axial_nodes, materials);
}
