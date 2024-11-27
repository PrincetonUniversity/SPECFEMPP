#include "IO/interface.hpp"
#include "source/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <boost/tokenizer.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

std::tuple<std::vector<std::shared_ptr<specfem::sources::source> >, type_real>
specfem::IO::read_sources(const std::string sources_file, const int nsteps,
                          const type_real user_t0, const type_real dt,
                          const specfem::simulation::type simulation_type) {

  const bool user_defined_start_time =
      (std::abs(user_t0) > std::numeric_limits<type_real>::epsilon());

  const specfem::wavefield::simulation_field source_wavefield_type =
      [&simulation_type]() -> specfem::wavefield::simulation_field {
    switch (simulation_type) {
    case specfem::simulation::type::forward:
      return specfem::wavefield::simulation_field::forward;
    case specfem::simulation::type::combined:
      return specfem::wavefield::simulation_field::backward;
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
