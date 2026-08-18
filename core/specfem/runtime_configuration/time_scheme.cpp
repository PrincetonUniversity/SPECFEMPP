#include "time_scheme.hpp"
#include "specfem/timescheme/newmark.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <ostream>

template <typename AssemblyFields>
std::shared_ptr<specfem::time_scheme::time_scheme>
specfem::runtime_configuration::time_scheme::instantiate(
    AssemblyFields &fields, const int nstep_between_samples,
    const specfem::simulation::type simulation_type) {

  std::shared_ptr<specfem::time_scheme::time_scheme> it;
  if (specfem::utilities::is_newmark_string(this->timescheme)) {
    if (simulation_type == specfem::simulation::type::forward) {

      it = std::make_shared<specfem::time_scheme::newmark<
          AssemblyFields, specfem::simulation::type::forward>>(
          fields, this->nstep, nstep_between_samples, this->dt, this->t0);
    } else if (simulation_type == specfem::simulation::type::combined) {
      it = std::make_shared<specfem::time_scheme::newmark<
          AssemblyFields, specfem::simulation::type::combined>>(
          fields, this->nstep, nstep_between_samples, this->dt, this->t0);
    } else if (simulation_type == specfem::simulation::type::combined_undoatt) {
      it = std::make_shared<specfem::time_scheme::newmark<
          AssemblyFields, specfem::simulation::type::combined_undoatt>>(
          fields, this->nstep, nstep_between_samples, this->dt, this->t0);
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

  return it;
}

specfem::runtime_configuration::time_scheme::time_scheme(
    const YAML::Node &timescheme) {

  try {
    const type_real t0 = [&timescheme]() -> type_real {
      if (timescheme["t0"]) {
        return -1.0 * timescheme["t0"].as<type_real>();
      } else {
        return 0.0;
      }
    }();

    *this = specfem::runtime_configuration::time_scheme(
        timescheme["type"].as<std::string>(), timescheme["dt"].as<type_real>(),
        timescheme["nstep"].as<int>(), t0);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading time marching timescheme. \n" << e.what();

    std::runtime_error(message.str());
  }
}

// Explicit template instantiations for dim2 and dim3 assembly fields
template std::shared_ptr<specfem::time_scheme::time_scheme>
specfem::runtime_configuration::time_scheme::instantiate<
    specfem::assembly::fields<specfem::element::dimension_tag::dim2>>(
    specfem::assembly::fields<specfem::element::dimension_tag::dim2> &fields,
    const int nstep_between_samples,
    const specfem::simulation::type simulation_type);

template std::shared_ptr<specfem::time_scheme::time_scheme>
specfem::runtime_configuration::time_scheme::instantiate<
    specfem::assembly::fields<specfem::element::dimension_tag::dim3>>(
    specfem::assembly::fields<specfem::element::dimension_tag::dim3> &fields,
    const int nstep_between_samples,
    const specfem::simulation::type simulation_type);
