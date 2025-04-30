#include "parameter_parser/writer/plot_wavefield.hpp"
#include "periodic_tasks/plot_wavefield.hpp"
#include "periodic_tasks/plotter.hpp"
#include "utilities/strings.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::plot_wavefield::plot_wavefield(
    const YAML::Node &Node) {

  const std::string output_format = [&]() -> std::string {
    if (Node["format"]) {
      return Node["format"].as<std::string>();
    } else {
      return "PNG";
    }
  }();

  if (specfem::utilities::is_onscreen_string(output_format)) {
    throw std::runtime_error("On screen plotting not supported");
  }

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

  const std::string component = [&]() -> std::string {
    if (Node["field"]) {
      return Node["field"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Plotting wavefield not specified in the display section");
    }
  }();

  const std::string wavefield_type = [&]() -> std::string {
    if (Node["simulation-field"]) {
      return Node["simulation-field"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Simulation field type not specified in the display section");
    }
  }();

  const int time_interval = [&]() -> int {
    if (Node["time-interval"]) {
      return Node["time-interval"].as<int>();
    } else {
      throw std::runtime_error(
          "Time interval not specified in the display section");
    }
  }();

  *this = specfem::runtime_configuration::plot_wavefield(
      output_format, output_folder, component, wavefield_type, time_interval);

  return;
}

std::shared_ptr<specfem::periodic_tasks::periodic_task>
specfem::runtime_configuration::plot_wavefield::instantiate_wavefield_plotter(
    const specfem::compute::assembly &assembly) const {

  const auto output_format = [&]() {
    if (specfem::utilities::is_png_string(this->output_format)) {
      return specfem::display::format::PNG;
    } else if (specfem::utilities::is_jpg_string(this->output_format)) {
      return specfem::display::format::JPG;
    } else if (specfem::utilities::is_onscreen_string(this->output_format)) {
      return specfem::display::format::on_screen;
    } else {
      throw std::runtime_error("Unknown plotter format");
    }
  }();

  const auto component = [&]() {
    if (specfem::utilities::is_displacement_string(this->component)) {
      return specfem::display::wavefield::displacement;
    } else if (specfem::utilities::is_velocity_string(this->component)) {
      return specfem::display::wavefield::velocity;
    } else if (specfem::utilities::is_acceleration_string(this->component)) {
      return specfem::display::wavefield::acceleration;
    } else if (specfem::utilities::is_pressure_string(this->component)) {
      return specfem::display::wavefield::pressure;
    } else if (specfem::utilities::is_rotation_string(this->component)) {
      return specfem::display::wavefield::rotation;
    } else {
      throw std::runtime_error(
          "Unknown wavefield component in the display section");
    }
  }();

  const auto wavefield = [&]() {
    if (specfem::utilities::is_forward_string(this->wavefield_type)) {
      return specfem::wavefield::simulation_field::forward;
    } else if (specfem::utilities::is_adjoint_string(this->wavefield_type)) {
      return specfem::wavefield::simulation_field::adjoint;
    } else if (specfem::utilities::is_backward_string(this->wavefield_type)) {
      return specfem::wavefield::simulation_field::backward;
    } else {
      throw std::runtime_error("Unknown wavefield type in the display section");
    }
  }();

  return std::make_shared<specfem::periodic_tasks::plot_wavefield>(
      assembly, output_format, component, wavefield, time_interval,
      this->output_folder);
}
