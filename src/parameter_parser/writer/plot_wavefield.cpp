#include "parameter_parser/writer/plot_wavefield.hpp"
#include "writer/plot_wavefield.hpp"
#include "writer/writer.hpp"
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
    if (Node["component"]) {
      return Node["component"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Wavefield component not specified in the display section");
    }
  }();

  const std::string wavefield_type = [&]() -> std::string {
    if (Node["wavefield_type"]) {
      return Node["wavefield_type"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Wavefield type not specified in the display section");
    }
  }();

  *this = specfem::runtime_configuration::plot_wavefield(
      output_format, output_folder, component, wavefield_type);

  return;
}

std::shared_ptr<specfem::writer::writer>
specfem::runtime_configuration::plot_wavefield::instantiate_wavefield_plotter(
    const specfem::compute::assembly &assembly) const {

  const auto output_format = [&]() {
    if (this->output_format == "PNG") {
      return specfem::display::format::PNG;
    } else if (this->output_format == "JPG") {
      return specfem::display::format::JPG;
    } else {
      throw std::runtime_error("Unknown plotter format");
    }
  }();

  const auto component = [&]() {
    if (this->component == "displacement_x") {
      return specfem::display::wavefield::displacement_x;
    } else if (this->component == "displacement_z") {
      return specfem::display::wavefield::displacement_z;
    } else if (this->component == "velocity_x") {
      return specfem::display::wavefield::velocity_x;
    } else if (this->component == "velocity_z") {
      return specfem::display::wavefield::velocity_z;
    } else if (this->component == "acceleration_x") {
      return specfem::display::wavefield::acceleration_x;
    } else if (this->component == "acceleration_z") {
      return specfem::display::wavefield::acceleration_z;
    } else if (this->component == "pressure") {
      return specfem::display::wavefield::pressure;
    } else {
      throw std::runtime_error(
          "Unknown wavefield component in the display section");
    }
  }();

  const auto wavefield = [&]() {
    if (this->wavefield_type == "forward") {
      return specfem::wavefield::type::forward;
    } else if (this->wavefield_type == "adjoint") {
      return specfem::wavefield::type::adjoint;
    } else {
      throw std::runtime_error("Unknown wavefield type in the display section");
    }
  }();

  return std::make_shared<specfem::writer::plot_wavefield>(
      assembly, output_format, component, wavefield, this->output_folder);
}
