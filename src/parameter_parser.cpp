#include "../include/parameter_parser.h"
#include "yaml-cpp/yaml.h"
#include <ctime>
#include <ostream>
#include <tuple>

specfem::TimeScheme::TimeScheme *
specfem::runtime_configuration::solver::instantiate() {
  specfem::TimeScheme::TimeScheme *it;

  throw std::runtime_error(
      "Could not instantiate solver : Error reading parameter file");

  return it;
};

specfem::TimeScheme::TimeScheme *
specfem::runtime_configuration::time_marching::instantiate() {

  specfem::TimeScheme::TimeScheme *it;
  if (this->timescheme == "Newmark") {
    it = new specfem::TimeScheme::Newmark(this->nstep, this->t0, this->dt);
  }

  return it;
}

std::tuple<specfem::quadrature::quadrature, specfem::quadrature::quadrature>
specfem::runtime_configuration::quadrature::instantiate() {
  specfem::quadrature::quadrature gllx(this->alpha, this->beta, this->ngllx);
  specfem::quadrature::quadrature gllz(this->alpha, this->beta, this->ngllz);
  return std::make_tuple(gllx, gllz);
}

specfem::runtime_configuration::header::header(const YAML::Node &Node) {
  *this = specfem::runtime_configuration::header(
      Node["title"].as<std::string>(), Node["description"].as<std::string>());
}

specfem::runtime_configuration::time_marching::time_marching(
    const YAML::Node &timescheme) {

  *this = specfem::runtime_configuration::time_marching(
      timescheme["type"].as<std::string>(), timescheme["dt"].as<type_real>(),
      timescheme["nstep"].as<int>());
}

specfem::runtime_configuration::run_setup::run_setup(const YAML::Node &Node) {
  *this = specfem::runtime_configuration::run_setup(
      Node["number-of-processors"].as<int>(), Node["number-of-runs"].as<int>());
}

specfem::runtime_configuration::quadrature::quadrature(const YAML::Node &Node) {
  *this = specfem::runtime_configuration::quadrature(
      Node["alpha"].as<type_real>(), Node["beta"].as<type_real>(),
      Node["ngllx"].as<int>(), Node["ngllz"].as<int>());
}

specfem::runtime_configuration::setup::setup(std::string parameter_file) {
  YAML::Node yaml = YAML::LoadFile(parameter_file);

  const YAML::Node &runtime_config = yaml["run-config"];
  const YAML::Node &simulation_setup = runtime_config["simulation-setup"];

  const YAML::Node &n_header = runtime_config["header"];
  const YAML::Node &n_solver = simulation_setup["solver"];
  const YAML::Node &n_quadrature = simulation_setup["quadrature"];
  const YAML::Node &n_run_setup = runtime_config["run-setup"];

  this->header = new specfem::runtime_configuration::header(n_header);

  if (n_solver["solver-type"].as<std::string>() == "time-marching") {
    const YAML::Node &n_timescheme = n_solver["Time-scheme"];
    this->solver =
        new specfem::runtime_configuration::time_marching(n_timescheme);
  } else {
    std::ostringstream message;
    message << "Error reading parameter file. \n"
            << "Solver = " << n_solver["solver-type"].as<std::string>()
            << "hasn't been implemented yet.";
    throw std::runtime_error(message.str());
  }

  this->run_setup = new specfem::runtime_configuration::run_setup(n_run_setup);
  this->quadrature =
      new specfem::runtime_configuration::quadrature(n_quadrature);
}

std::string specfem::runtime_configuration::setup::print_header() {

  std::ostringstream message;
  // current date/time based on current system
  time_t now = time(0);

  // convert now to string form
  char *dt = ctime(&now);

  message << "SPECFEM2D SIMULATION\n"
          << "--------------------\n"
          << "Title : " << this->header->get_title() << "\n"
          << "Discription: " << this->header->get_description() << "\n"
          << "Simulation start time: " << dt;

  return message.str();
}
