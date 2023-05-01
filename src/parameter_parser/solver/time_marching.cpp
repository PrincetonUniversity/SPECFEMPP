#include "parameter_parser/interface.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"

specfem::TimeScheme::TimeScheme *
specfem::runtime_configuration::solver::time_marching::instantiate(
    const int nstep_between_samples) {

  specfem::TimeScheme::TimeScheme *it;
  if (this->timescheme == "Newmark") {
    it = new specfem::TimeScheme::Newmark(this->nstep, this->t0, this->dt,
                                          nstep_between_samples);
  }

  return it;
}

specfem::runtime_configuration::solver::time_marching::time_marching(
    const YAML::Node &timescheme) {

  *this = specfem::runtime_configuration::solver::time_marching(
      timescheme["type"].as<std::string>(), timescheme["dt"].as<type_real>(),
      timescheme["nstep"].as<int>());
}
