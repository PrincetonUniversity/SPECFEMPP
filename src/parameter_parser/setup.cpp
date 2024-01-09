#include "parameter_parser/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <ostream>
#include <sys/stat.h>
#include <tuple>

void create_folder_if_not_exists(const std::string &folder_name) {
  struct stat info;

  if (stat(folder_name.c_str(), &info) != 0) {
    std::cout << "Creating folder: " << folder_name << std::endl;
    mkdir(folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  } else if (info.st_mode & S_IFDIR) {
    std::cout << "Folder already exists: " << folder_name << std::endl;
  } else {
    std::cout << "Error: " << folder_name << " is not a directory" << std::endl;
  }
}

specfem::runtime_configuration::setup::setup(const std::string &parameter_file,
                                             const std::string &default_file) {
  YAML::Node parameter_yaml = YAML::LoadFile(parameter_file);
  YAML::Node default_yaml = YAML::LoadFile(default_file);

  const YAML::Node &runtime_config = parameter_yaml["parameters"];
  const YAML::Node &default_config = default_yaml["default-parameters"];

  const YAML::Node &simulation_setup = runtime_config["simulation-setup"];
  const YAML::Node &n_solver = simulation_setup["solver"];

  try {
    this->header = std::make_unique<specfem::runtime_configuration::header>(
        runtime_config["header"]);
  } catch (YAML::InvalidNode &e) {
    std::ostringstream message;
    message << "Error reading specfem parameter header. \n" << e.what();

    throw std::runtime_error(message.str());
  }

  if (const YAML::Node &n_quadrature = simulation_setup["quadrature"]) {
    this->quadrature =
        std::make_unique<specfem::runtime_configuration::quadrature>(
            n_quadrature);
  } else if (const YAML::Node &n_quadrature = default_config["quadrature"]) {
    this->quadrature =
        std::make_unique<specfem::runtime_configuration::quadrature>(
            n_quadrature);
  } else {
    throw std::runtime_error("Error reading specfem quadrature config.");
  }

  if (const YAML::Node &n_run_setup = runtime_config["run-setup"]) {
    this->run_setup =
        std::make_unique<specfem::runtime_configuration::run_setup>(
            n_run_setup);
  } else if (const YAML::Node &n_run_setup = default_yaml["run-setup"]) {
    this->run_setup =
        std::make_unique<specfem::runtime_configuration::run_setup>(
            n_run_setup);
  } else {
    throw std::runtime_error("Error reading specfem runtime configuration.");
  }

  try {
    this->databases = std::make_unique<
        specfem::runtime_configuration::database_configuration>(
        runtime_config["databases"]);
  } catch (YAML::InvalidNode &e) {
    std::ostringstream message;

    message << "Error reading specfem database configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }

  try {
    this->receivers =
        std::make_unique<specfem::runtime_configuration::receivers>(
            runtime_config["receivers"]);
  } catch (YAML::InvalidNode &e) {
    std::ostringstream message;
    message << "Error reading specfem receiver configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }

  try {
    this->seismogram =
        std::make_unique<specfem::runtime_configuration::seismogram>(
            runtime_config["seismogram"]);
  } catch (YAML::InvalidNode &e) {
    YAML::Node seismogram;
    seismogram["seismogram-format"] = "ascii";
    std::string folder_name = "results";
    create_folder_if_not_exists(folder_name);
    seismogram["output-folder"] = folder_name;
    this->seismogram =
        std::make_unique<specfem::runtime_configuration::seismogram>(
            seismogram);
  }

  try {
    const YAML::Node &n_time_marching = n_solver["time-marching"];
    const YAML::Node &n_timescheme = n_time_marching["time-scheme"];

    this->solver =
        std::make_unique<specfem::runtime_configuration::solver::time_marching>(
            n_timescheme);
  } catch (YAML::InvalidNode &e) {
    std::ostringstream message;
    message << "Error reading specfem solver configuration. \n" << e.what();
    throw std::runtime_error(message.str());
  }
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
