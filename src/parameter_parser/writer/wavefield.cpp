#include "parameter_parser/writer/wavefield.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/reader.hpp"
#include "periodic_tasks/wavefield_reader.hpp"
#include "periodic_tasks/wavefield_writer.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::wavefield::wavefield(
    const YAML::Node &Node, const specfem::simulation::type type) {

  const std::string output_format = [&]() -> std::string {
    if (Node["format"]) {
      return Node["format"].as<std::string>();
    } else {
      return "ASCII";
    }
  }();

  const std::string output_folder = [&]() -> std::string {
    if (Node["directory"]) {
      return Node["directory"].as<std::string>();
    } else {
      return boost::filesystem::current_path().string();
    }
  }();

  if (!boost::filesystem::is_directory(
          boost::filesystem::path(output_folder))) {
    std::ostringstream message;
    message << "Output folder : " << output_folder << " does not exist.";
    throw std::runtime_error(message.str());
  }

  const int time_interval = [&]() -> int {
    if (Node["time_interval"]) {
      return Node["time_interval"].as<int>();
    } else {
      return 0;
    }
  }();

  const std::string time_interval_by_memory = [&]() -> std::string {
    if (Node["time_interval_by_memory"]) {
      if (time_interval != 0) {
        throw std::runtime_error(
            "time_interval and time_interval_by_memory cannot be used "
            "simultaneously");
      }
      return Node["time_interval_by_memory"].as<std::string>();
    } else {
      return "";
    }
  }();

  const bool include_last_step = [&]() -> bool {
    if (Node["include_last_step"]) {
      return Node["include_last_step"].as<bool>();
    } else {
      return true;
    }
  }();

  if (time_interval == 0 && !include_last_step) {
    std::ostringstream message;
    message << "************************************************\n"
            << "Warning : Wavefield writer does not write any wavefield. \n"
            << "************************************************\n";
    std::cout << message.str();
  }

  *this = specfem::runtime_configuration::wavefield(
      output_format, output_folder, type, time_interval,
      time_interval_by_memory, include_last_step);

  return;
}

std::shared_ptr<specfem::periodic_tasks::periodic_task>
specfem::runtime_configuration::wavefield::instantiate_wavefield_writer()
    const {

  const std::shared_ptr<specfem::periodic_tasks::periodic_task> writer =
      [&]() -> std::shared_ptr<specfem::periodic_tasks::periodic_task> {
    if (this->simulation_type == specfem::simulation::type::forward) {
      if (this->output_format == "HDF5") {
        return std::make_shared<
            specfem::periodic_tasks::wavefield_writer<specfem::io::HDF5> >(
            this->output_folder, this->time_interval, this->include_last_step);
      } else if (this->output_format == "ASCII") {
        return std::make_shared<
            specfem::periodic_tasks::wavefield_writer<specfem::io::ASCII> >(
            this->output_folder, this->time_interval, this->include_last_step);
      } else {
        throw std::runtime_error("Unknown wavefield format");
      }
    } else {
      return nullptr;
    }
  }();

  return writer;
}

std::shared_ptr<specfem::periodic_tasks::periodic_task>
specfem::runtime_configuration::wavefield::instantiate_wavefield_reader()
    const {

  const std::shared_ptr<specfem::periodic_tasks::periodic_task> reader =
      [&]() -> std::shared_ptr<specfem::periodic_tasks::periodic_task> {
    if (this->simulation_type == specfem::simulation::type::combined) {
      if (this->output_format == "HDF5") {
        return std::make_shared<
            specfem::periodic_tasks::wavefield_reader<specfem::io::HDF5> >(
            this->output_folder, this->time_interval, this->include_last_step);
      } else if (this->output_format == "ASCII") {
        return std::make_shared<
            specfem::periodic_tasks::wavefield_reader<specfem::io::ASCII> >(
            this->output_folder, this->time_interval, this->include_last_step);
      } else {
        throw std::runtime_error("Unknown wavefield format");
      }
    } else {
      return nullptr;
    }
  }();

  return reader;
}
