#include "parameter_parser/interface.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"

specfem::TimeScheme::TimeScheme *
specfem::runtime_configuration::solver::solver::instantiate(
    const int nstep_between_samples) {
  specfem::TimeScheme::TimeScheme *it = NULL;

  throw std::runtime_error(
      "Could not instantiate solver : Error reading parameter file");

  return it;
};
