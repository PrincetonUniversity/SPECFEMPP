#include "parameter_parser/interface.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <ostream>

std::shared_ptr<specfem::TimeScheme::TimeScheme>
specfem::runtime_configuration::solver::time_marching::instantiate(
    const int nstep_between_samples) {

  std::shared_ptr<specfem::TimeScheme::TimeScheme> it;
  if (this->timescheme == "Newmark") {
    it = std::make_shared<specfem::TimeScheme::Newmark>(
        this->nstep, this->t0, this->dt, nstep_between_samples);
  }

  // User output
  std::cout << *it << "\n";

  return it;
}

specfem::runtime_configuration::solver::time_marching::time_marching(
    const YAML::Node &timescheme) {

  try {
    *this = specfem::runtime_configuration::solver::time_marching(
        timescheme["type"].as<std::string>(), timescheme["dt"].as<type_real>(),
        timescheme["nstep"].as<int>());
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading time marching timescheme. \n" << e.what();

    std::runtime_error(message.str());
  }
}
