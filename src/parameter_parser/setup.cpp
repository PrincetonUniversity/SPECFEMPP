#include "parameter_parser/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <chrono>
#include <ctime>
#include <ostream>
#include <tuple>

specfem::runtime_configuration::setup::setup(std::string parameter_file) {
  YAML::Node yaml = YAML::LoadFile(parameter_file);

  const YAML::Node &runtime_config = yaml["parameters"];
  const YAML::Node &simulation_setup = runtime_config["simulation-setup"];

  const YAML::Node &n_header = runtime_config["header"];
  const YAML::Node &n_solver = simulation_setup["solver"];
  const YAML::Node &n_quadrature = simulation_setup["quadrature"];
  const YAML::Node &n_run_setup = runtime_config["run-setup"];
  const YAML::Node &n_databases = runtime_config["databases"];
  const YAML::Node &n_seismogram = runtime_config["seismogram"];

  this->header = new specfem::runtime_configuration::header(n_header);

  try {
    const YAML::Node &n_time_marching = n_solver["time-marching"];
    const YAML::Node &n_timescheme = n_time_marching["time-scheme"];

    this->solver =
        new specfem::runtime_configuration::solver::time_marching(n_timescheme);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;
    message << "Error reading parameter file. \n"
            << "Solver = " << n_solver["solver-type"].as<std::string>()
            << "hasn't been implemented yet.\n"
            << e.what();
    throw std::runtime_error(message.str());
  }

  this->run_setup = new specfem::runtime_configuration::run_setup(n_run_setup);
  this->quadrature =
      new specfem::runtime_configuration::quadrature(n_quadrature);

  this->databases =
      new specfem::runtime_configuration::database_configuration(n_databases);

  this->seismogram =
      new specfem::runtime_configuration::seismogram(n_seismogram);
}

std::string specfem::runtime_configuration::setup::print_header(
    std::chrono::time_point<std::chrono::high_resolution_clock> now) {

  std::ostringstream message;

  // convert now to string form
  std::time_t c_now = std::chrono::system_clock::to_time_t(now);

  message << "================================================\n"
          << "              SPECFEM2D SIMULATION\n"
          << "================================================\n\n"
          << "Title : " << this->header->get_title() << "\n"
          << "Discription: " << this->header->get_description() << "\n"
          << "Simulation start time: " << ctime(&c_now)
          << "------------------------------------------------\n";

  return message.str();
}
