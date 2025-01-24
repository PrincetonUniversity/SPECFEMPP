#include "parameter_parser/time_scheme/time_scheme.hpp"
#include "timescheme/newmark.hpp"
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
    } else if (this->type == specfem::simulation::type::combined) {
      it = std::make_shared<
          specfem::time_scheme::newmark<specfem::simulation::type::combined> >(
          this->nstep, nstep_between_samples, this->dt, this->t0);
    } else {
      std::ostringstream message;
      message << "Error in time scheme instantiation. \n"
              << "Unknown simulation type.";
      throw std::runtime_error(message.str());
    }
  } else {
    std::ostringstream message;
    message << "Error in time scheme instantiation. \n"
            << "Unknown time scheme.";
    throw std::runtime_error(message.str());
  }

  // User output
  // std::cout << *it << "\n";

  return it;
}

specfem::runtime_configuration::time_scheme::time_scheme::time_scheme(
    const YAML::Node &timescheme, const specfem::simulation::type simulation) {

  try {
    const type_real t0 = [&timescheme]() -> type_real {
      if (timescheme["t0"]) {
        return -1.0 * timescheme["t0"].as<type_real>();
      } else {
        return 0.0;
      }
    }();

    *this = specfem::runtime_configuration::time_scheme::time_scheme(
        timescheme["type"].as<std::string>(), timescheme["dt"].as<type_real>(),
        timescheme["nstep"].as<int>(), t0, simulation);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading time marching timescheme. \n" << e.what();

    std::runtime_error(message.str());
  }
}
