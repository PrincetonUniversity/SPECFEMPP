#include "parameter_parser/time_scheme/time_scheme.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <ostream>

std::shared_ptr<specfem::time_scheme::time_scheme>
specfem::runtime_configuration::time_scheme::time_scheme::instantiate(
    const int nstep_between_samples) {

  std::shared_ptr<specfem::time_scheme::time_scheme> it;
  if (this->timescheme == "Newmark") {
    if (this->type == specfem::simulation::type::forward) {

      it = std::make_shared<
          specfem::time_scheme::newmark<specfem::simulation::type::forward> >(
          this->nstep, nstep_between_samples, this->dt, this->t0);
    } else {
      throw std::runtime_error(
          "Could not instantiate solver : Wrong simulation time");
    }
  }

  // User output
  std::cout << *it << "\n";

  return it;
}

specfem::runtime_configuration::time_scheme::time_scheme::time_scheme(
    const YAML::Node &timescheme, const specfem::simulation::type simulation) {

  try {
    *this = specfem::runtime_configuration::time_scheme::time_scheme(
        timescheme["type"].as<std::string>(), timescheme["dt"].as<type_real>(),
        timescheme["nstep"].as<int>(), simulation);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading time marching timescheme. \n" << e.what();

    std::runtime_error(message.str());
  }
}
