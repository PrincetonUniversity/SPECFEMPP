// Internal Includes
#include "mesh/mesh.hpp"
#include "IO/fortranio/interface.hpp"
#include "IO/interface.hpp"
#include "IO/mesh/impl/fortran/read_boundaries.hpp"
#include "IO/mesh/impl/fortran/read_elements.hpp"
#include "IO/mesh/impl/fortran/read_interfaces.hpp"
#include "IO/mesh/impl/fortran/read_material_properties.hpp"
#include "IO/mesh/impl/fortran/read_mesh_database.hpp"
#include "IO/mesh/impl/fortran/read_parameters.hpp"
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
specfem::IO::read_mesh(const std::string filename,
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
        specfem::IO::mesh::impl::fortran::read_mesh_database_header(stream,
                                                                    mpi);
    mesh.nspec = nspec;
    mesh.npgeo = npgeo;
    mesh.nproc = nproc;
  } catch (std::runtime_error &e) {
    throw;
  }

  // Mesh class to be populated from the database file.
  try {
    mesh.control_nodes.coord =
        specfem::IO::mesh::impl::fortran::read_coorg_elements(stream,
                                                              mesh.npgeo, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.parameters =
        specfem::IO::mesh::impl::fortran::read_mesh_parameters(stream, mpi);
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
        specfem::IO::mesh::impl::fortran::read_mesh_database_attenuation(stream,
                                                                         mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.materials = specfem::IO::mesh::impl::fortran::read_material_properties(
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
    mesh.boundaries = specfem::IO::mesh::impl::fortran::read_boundaries(
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
        specfem::IO::mesh::impl::fortran::read_coupled_interfaces(
            stream, mesh.parameters.num_fluid_solid_edges,
            mesh.parameters.num_fluid_poro_edges,
            mesh.parameters.num_solid_poro_edges, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.tangential_nodes =
        specfem::IO::mesh::impl::fortran::read_tangential_elements(
            stream, mesh.parameters.nnodes_tangential_curve);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.axial_nodes = specfem::IO::mesh::impl::fortran::read_axial_elements(
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

  const auto l_elastic_sv_isotropic =
      mesh.materials.elastic_sv_isotropic.material_properties;
  const auto l_acoustic_isotropic =
      mesh.materials.acoustic_isotropic.material_properties;

  const auto l_elastic_sv_anisotropic =
      mesh.materials.elastic_sv_anisotropic.material_properties;

  for (const auto material : l_elastic_sv_isotropic) {
    mpi->cout(material.print());
  }

  for (const auto material : l_acoustic_isotropic) {
    mpi->cout(material.print());
  }

  for (const auto material : l_elastic_sv_anisotropic) {
    mpi->cout(material.print());
  }

  assert(l_elastic_sv_isotropic.size() + l_acoustic_isotropic.size() +
             l_elastic_sv_anisotropic.size() ==
         mesh.materials.n_materials);

  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
      mesh.materials, mesh.boundaries);

  return mesh;
}
