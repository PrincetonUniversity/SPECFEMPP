#include "source/external.hpp"
#include "source/force_source.hpp"
#include "source/interface.hpp"
#include "source/moment_tensor_source.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <boost/tokenizer.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

std::tuple<std::vector<std::shared_ptr<specfem::sources::source> >, type_real>
specfem::sources::read_sources(
    const std::string sources_file, const int nsteps, const type_real dt,
    const specfem::simulation::type simulation_type) {

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
  for (auto N : Node) {
    if (YAML::Node force_source = N["force"]) {
      sources.push_back(std::make_shared<specfem::sources::force>(
          force_source, nsteps, dt, source_wavefield_type));
    } else if (YAML::Node moment_tensor_source = N["moment-tensor"]) {
      sources.push_back(std::make_shared<specfem::sources::moment_tensor>(
          moment_tensor_source, nsteps, dt, source_wavefield_type));
    } else if (YAML::Node external = N["user-defined"]) {
      sources.push_back(std::make_shared<specfem::sources::external>(
          external, nsteps, dt, source_wavefield_type));
    } else if (YAML::Node adjoint_source = N["adjoint-source"]) {
      sources.push_back(std::make_shared<specfem::sources::adjoint_source>(
          adjoint_source, nsteps, dt));
    } else {
      throw std::runtime_error("Unknown source type");
    }
  }

  if (sources.size() != nsources) {
    std::ostringstream message;
    message << "Found only " << sources.size()
            << " number of sources. Total number of sources in " << sources_file
            << " are" << nsources
            << " Please check if there is a error in sources file.";
    throw std::runtime_error(message.str());
  }

  type_real t0 = std::numeric_limits<type_real>::max();
  for (auto &source : sources) {
    type_real cur_t0 = source->get_t0();
    std::cout << cur_t0 << std::endl;
    if (cur_t0 < t0) {
      t0 = cur_t0;
    }
  }

  for (auto &source : sources) {
    type_real cur_t0 = source->get_t0();
    source->update_tshift(cur_t0 - t0);
  }

  return std::make_tuple(sources, t0);
}
