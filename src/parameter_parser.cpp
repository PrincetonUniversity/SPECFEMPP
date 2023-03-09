#include "../include/parameter_parser.h"
#include "../include/globals.h"
#include "../include/writer.h"
#include "yaml-cpp/yaml.h"
#include <boost/filesystem.hpp>
#include <chrono>
#include <ctime>
#include <ostream>
#include <tuple>

specfem::TimeScheme::TimeScheme *
specfem::runtime_configuration::solver::instantiate(
    const int nstep_between_samples) {
  specfem::TimeScheme::TimeScheme *it = NULL;

  throw std::runtime_error(
      "Could not instantiate solver : Error reading parameter file");

  return it;
};

specfem::TimeScheme::TimeScheme *
specfem::runtime_configuration::time_marching::instantiate(
    const int nstep_between_samples) {

  specfem::TimeScheme::TimeScheme *it;
  if (this->timescheme == "Newmark") {
    it = new specfem::TimeScheme::Newmark(this->nstep, this->t0, this->dt,
                                          nstep_between_samples);
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

specfem::runtime_configuration::seismogram::seismogram(
    const YAML::Node &seismogram) {

  boost::filesystem::path cwd = boost::filesystem::current_path();
  std::string output_folder = cwd.string();
  if (seismogram["output-folder"]) {
    output_folder = seismogram["output-folder"].as<std::string>();
  }

  if (!boost::filesystem::is_directory(
          boost::filesystem::path(output_folder))) {
    std::ostringstream message;
    message << "Output folder : " << output_folder << " does not exist.";
    throw std::runtime_error(message.str());
  }

  const int nstep_between_samples =
      seismogram["nstep_between_samples"].as<int>();

  *this = specfem::runtime_configuration::seismogram(
      seismogram["stations-file"].as<std::string>(),
      seismogram["angle"].as<type_real>(),
      seismogram["nstep_between_samples"].as<int>(),
      seismogram["seismogram-format"].as<std::string>(), output_folder);

  // Allocate seismogram types
  assert(seismogram["seismogram-type"].IsSequence());

  for (YAML::Node seismogram_type : seismogram["seismogram-type"]) {
    if (seismogram_type.as<std::string>() == "displacement") {
      this->stypes.push_back(specfem::seismogram::displacement);
    } else if (seismogram_type.as<std::string>() == "velocity") {
      this->stypes.push_back(specfem::seismogram::velocity);
    } else if (seismogram_type.as<std::string>() == "acceleration") {
      this->stypes.push_back(specfem::seismogram::velocity);
    } else {
      std::runtime_error("Seismograms config could not be read properly");
    }
  }

  return;
}

specfem::writer::writer *
specfem::runtime_configuration::seismogram::instantiate_seismogram_writer(
    std::vector<specfem::receivers::receiver *> &receivers,
    specfem::compute::receivers *compute_receivers, const type_real dt,
    const type_real t0) const {

  specfem::seismogram::format::type type;
  if (this->seismogram_format == "seismic_unix" ||
      this->seismogram_format == "su") {
    type = specfem::seismogram::format::seismic_unix;
  } else if (this->seismogram_format == "ascii") {
    type = specfem::seismogram::format::ascii;
  }

  specfem::writer::writer *writer = new specfem::writer::seismogram(
      receivers, compute_receivers, type, this->output_folder, dt, t0,
      this->nstep_between_samples);

  return writer;
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

specfem::runtime_configuration::database_configuration::database_configuration(
    const YAML::Node &Node) {
  *this = specfem::runtime_configuration::database_configuration(
      Node["mesh-database"].as<std::string>(),
      Node["source-file"].as<std::string>());
}

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
        new specfem::runtime_configuration::time_marching(n_timescheme);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;
    message << "Error reading parameter file. \n"
            << "Solver = " << n_solver["solver-type"].as<std::string>()
            << "hasn't been implemented yet.\n"
            << e.what();
    throw std::runtime_error(message.str());
  }

  // if (n_solver["solver-type"].as<std::string>() == "time-marching") {
  //   const YAML::Node &n_timescheme = n_solver["Time-scheme"];
  //   this->solver =
  //       new specfem::runtime_configuration::time_marching(n_timescheme);
  // } else {
  //   std::ostringstream message;
  //   message << "Error reading parameter file. \n"
  //           << "Solver = " << n_solver["solver-type"].as<std::string>()
  //           << "hasn't been implemented yet.";
  //   throw std::runtime_error(message.str());
  // }

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
