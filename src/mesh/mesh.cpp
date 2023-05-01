#include "mesh/mesh.hpp"
#include "compute/interface.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>
#include <vector>

specfem::mesh::mesh::mesh(const std::string filename,
                          std::vector<specfem::material::material *> &materials,
                          const specfem::MPI::MPI *mpi) {

  std::ifstream stream;
  stream.open(filename);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open database file");
  }

  try {
    auto [nspec, npgeo, nproc] =
        specfem::mesh::IO::fortran::read_mesh_database_header(stream, mpi);
    this->nspec = nspec;
    this->npgeo = npgeo;
    this->nproc = nproc;
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->coorg = specfem::mesh::IO::fortran::read_coorg_elements(
        stream, this->npgeo, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->parameters = specfem::mesh::properties(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  int nspec_all = mpi->reduce(this->parameters.nspec, specfem::MPI::sum);
  int nelem_acforcing_all =
      mpi->reduce(this->parameters.nelem_acforcing, specfem::MPI::sum);
  int nelem_acoustic_surface_all =
      mpi->reduce(this->parameters.nelem_acoustic_surface, specfem::MPI::sum);

  // std::ostringstream message;
  // message << "Number of spectral elements . . . . . . . . . .(nspec) = "
  //         << nspec_all
  //         << "\n"
  //            "Number of control nodes per element . . . . . .(NGNOD) = "
  //         << this->parameters.ngnod
  //         << "\n"
  //            "Number of points for display . . . . . . .(pointsdisp) = "
  //         << this->parameters.pointsdisp
  //         << "\n"
  //            "Number of element material sets . . . . . . . .(numat) = "
  //         << this->parameters.numat
  //         << "\n"
  //            "Number of acoustic forcing elements .(nelem_acforcing) = "
  //         << nelem_acforcing_all
  //         << "\n"
  //            "Number of acoustic free surf .(nelem_acoustic_surface) = "
  //         << nelem_acoustic_surface_all;

  // mpi->cout(message.str());

  try {
    auto [n_sls, attenuation_f0_reference, read_velocities_at_f0] =
        specfem::mesh::IO::fortran::read_mesh_database_attenuation(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    materials = specfem::mesh::IO::fortran::read_material_properties(
        stream, this->parameters.numat, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->material_ind =
        specfem::mesh::material_ind(stream, this->parameters.ngnod, this->nspec,
                                    this->parameters.numat, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->interface = specfem::mesh::interfaces::interface(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->abs_boundary = specfem::mesh::boundaries::absorbing_boundary(
        stream, this->parameters.nelemabs, this->parameters.nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->acforcing_boundary = specfem::mesh::boundaries::forcing_boundary(
        stream, this->parameters.nelem_acforcing, this->parameters.nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->acfree_surface = specfem::mesh::surfaces::acoustic_free_surface(
        stream, this->parameters.nelem_acoustic_surface, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    specfem::mesh::IO::fortran::read_mesh_database_coupled(
        stream, this->parameters.num_fluid_solid_edges,
        this->parameters.num_fluid_poro_edges,
        this->parameters.num_solid_poro_edges, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->tangential_nodes = specfem::mesh::elements::tangential_elements(
        stream, this->parameters.nnodes_tangential_curve);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->axial_nodes = specfem::mesh::elements::axial_elements(
        stream, this->parameters.nelem_on_the_axis, this->nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // Check if database file was read completely
  if (stream.get() && !stream.eof()) {
    throw std::runtime_error("The Database file wasn't fully read. Is there "
                             "anything written after axial elements?");
  }

  stream.close();

  return;
}

std::string specfem::mesh::mesh::print(
    std::vector<specfem::material::material *> materials) const {

  int n_elastic = 0;
  int n_acoustic = 0;

  Kokkos::parallel_reduce(
      "setup_compute::properties_ispec", specfem::kokkos::HostRange(0, nspec),
      [=](const int ispec, int &l_elastic, int &l_acoustic) {
        const int imat = this->material_ind.kmato(ispec);
        if (materials[imat]->get_ispec_type() == specfem::elements::elastic) {
          l_elastic++;
        } else if (materials[imat]->get_ispec_type() ==
                   specfem::elements::acoustic) {
          l_acoustic++;
        }
      },
      n_elastic, n_acoustic);

  std::ostringstream message;

  message
      << "Spectral element information:\n"
      << "------------------------------\n"
      << "Total number of spectral elements : " << this->nspec << "\n"
      << "Total number of spectral elements assigned to elastic material : "
      << n_elastic << "\n"
      << "Total number of spectral elements assigned to acoustic material : "
      << n_acoustic << "\n"
      << "Total number of geometric points : " << this->npgeo << "\n";

  return message.str();
}
