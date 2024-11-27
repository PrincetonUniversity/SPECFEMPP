// Internal Includes
#include "IO/interface.hpp"
#include "IO/fortranio/interface.hpp"
#include "IO/mesh/impl/fortran/read_boundaries.hpp"
#include "IO/mesh/impl/fortran/read_elements.hpp"
#include "IO/mesh/impl/fortran/read_interfaces.hpp"
#include "IO/mesh/impl/fortran/read_material_properties.hpp"
#include "IO/mesh/impl/fortran/read_mesh_database.hpp"
#include "IO/mesh/impl/fortran/read_properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "material/material.hpp"
#include "mesh/mesh.hpp"
#include "mesh/tags/tags.hpp"
#include "receiver/interface.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"

// External/Standard Libraries
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

std::tuple<std::vector<std::shared_ptr<specfem::sources::source> >, type_real>
specfem::IO::read_sources(const std::string sources_file, const int nsteps,
                          const type_real user_t0, const type_real dt,
                          const specfem::simulation::type simulation_type) {

  const bool user_defined_start_time =
      (std::abs(user_t0) > std::numeric_limits<type_real>::epsilon());

  const specfem::wavefield::type source_wavefield_type =
      [&simulation_type]() -> specfem::wavefield::type {
    switch (simulation_type) {
    case specfem::simulation::type::forward:
      return specfem::wavefield::type::forward;
    case specfem::simulation::type::combined:
      return specfem::wavefield::type::backward;
    default:
      throw std::runtime_error("Unknown simulation type");
    }
  }();

  // read sources file
  std::vector<std::shared_ptr<specfem::sources::source> > sources;
  YAML::Node yaml = YAML::LoadFile(sources_file);
  int nsources = yaml["number-of-sources"].as<int>();
  YAML::Node Node = yaml["sources"];
  assert(Node.IsSequence());

  // Note: Make sure you name the YAML node different from the name of the
  // source class Otherwise, the compiler will get confused and throw an error
  // I've spent hours debugging this issue. It is very annoying since it only
  // shows up on CUDA compiler
  int number_of_sources = 0;
  int number_of_adjoint_sources = 0;
  for (auto N : Node) {
    if (YAML::Node force_source = N["force"]) {
      sources.push_back(std::make_shared<specfem::sources::force>(
          force_source, nsteps, dt, source_wavefield_type));
      number_of_sources++;
    } else if (YAML::Node moment_tensor_source = N["moment-tensor"]) {
      sources.push_back(std::make_shared<specfem::sources::moment_tensor>(
          moment_tensor_source, nsteps, dt, source_wavefield_type));
      number_of_sources++;
    } else if (YAML::Node external_source = N["user-defined"]) {
      sources.push_back(std::make_shared<specfem::sources::external>(
          external_source, nsteps, dt, source_wavefield_type));
      number_of_sources++;
    } else if (YAML::Node adjoint_node = N["adjoint-source"]) {
      if (!adjoint_node["station_name"] || !adjoint_node["network_name"]) {
        throw std::runtime_error(
            "Station name and network name are required for adjoint source");
      }
      sources.push_back(std::make_shared<specfem::sources::adjoint_source>(
          adjoint_node, nsteps, dt));
      number_of_adjoint_sources++;
    } else {
      throw std::runtime_error("Unknown source type");
    }
  }

  if (number_of_sources == 0) {
    throw std::runtime_error("No sources found in the sources file");
  }

  if (simulation_type == specfem::simulation::type::combined &&
      number_of_adjoint_sources == 0) {
    throw std::runtime_error("No adjoint sources found in the sources file");
  }

  if (simulation_type == specfem::simulation::type::forward &&
      number_of_adjoint_sources > 0) {
    throw std::runtime_error("Adjoint sources found in the sources file for "
                             "forward simulation");
  }

  if (sources.size() != nsources) {
    std::ostringstream message;
    message << "Found only " << sources.size()
            << " number of sources. Total number of sources in " << sources_file
            << " are" << nsources
            << " Please check if there is a error in sources file.";
    throw std::runtime_error(message.str());
  }

  type_real min_t0 = std::numeric_limits<type_real>::max();
  type_real min_tshift = std::numeric_limits<type_real>::max();
  for (auto &source : sources) {
    type_real cur_t0 = source->get_t0();
    type_real cur_tshift = source->get_tshift();
    if (cur_t0 < min_t0) {
      min_t0 = cur_t0;
    }
    if (cur_tshift < min_tshift) {
      min_tshift = cur_tshift;
    }
  }

  type_real t0;
  if (user_defined_start_time) {
    if (user_t0 > min_t0 - min_tshift)
      throw std::runtime_error("User defined start time is less than minimum "
                               "required for stability");

    t0 = user_t0;
  } else {
    // Update tshift for auto detected start time
    for (auto &source : sources) {
      type_real cur_t0 = source->get_t0();
      source->update_tshift(cur_t0 - min_t0);
    }

    t0 = min_t0;
  }

  return std::make_tuple(sources, t0);
}

std::vector<std::shared_ptr<specfem::receivers::receiver> >
specfem::IO::read_receivers(const std::string stations_file,
                            const type_real angle) {

  boost::char_separator<char> sep(" ");
  std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers;
  std::fstream stations;
  stations.open(stations_file, std::ios::in);
  if (stations.is_open()) {
    std::string line;
    // Read stations file line by line
    while (std::getline(stations, line)) {
      // split every line with " " delimiter
      boost::tokenizer<boost::char_separator<char> > tokens(line, sep);
      std::vector<std::string> current_station;
      for (const auto &t : tokens) {
        current_station.push_back(t);
      }
      // check if the read line meets the format
      assert(current_station.size() == 6);
      // get the x and z coordinates of the station;
      const std::string network_name = current_station[0];
      const std::string station_name = current_station[1];
      const type_real x = static_cast<type_real>(std::stod(current_station[2]));
      const type_real z = static_cast<type_real>(std::stod(current_station[3]));

      receivers.push_back(std::make_shared<specfem::receivers::receiver>(
          network_name, station_name, x, z, angle));
    }

    stations.close();
  }

  return receivers;
}

specfem::mesh::mesh specfem::IO::read_mesh(const std::string filename,
                                           const specfem::MPI::MPI *mpi) {

  // Declaring empty mesh objects
  specfem::mesh::mesh mesh;

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
        specfem::IO::mesh::impl::fortran::read_properties(stream, mpi);
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

  const auto l_elastic_isotropic =
      mesh.materials.elastic_isotropic.material_properties;
  const auto l_acoustic_isotropic =
      mesh.materials.acoustic_isotropic.material_properties;

  for (const auto material : l_elastic_isotropic) {
    mpi->cout(material.print());
  }

  for (const auto material : l_acoustic_isotropic) {
    mpi->cout(material.print());
  }

  assert(l_elastic_isotropic.size() + l_acoustic_isotropic.size() ==
         mesh.materials.n_materials);

  mesh.tags = specfem::mesh::tags(mesh.materials, mesh.boundaries);

  return mesh;
}
