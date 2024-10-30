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

  const std::string wavefield_type = [&]() -> std::string {
    if (Node["type"]) {
      return Node["type"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Display type not specified in plotter configuration");
    }
  }();

  *this = specfem::runtime_configuration::plot_wavefield(
      output_format, output_folder, wavefield_type);

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

  const auto wavefield_type = [&]() {
    if (this->wavefield_type == "displacement") {
      return specfem::display::wavefield::displacement;
    } else if (this->wavefield_type == "velocity") {
      return specfem::display::wavefield::velocity;
    } else if (this->wavefield_type == "acceleration") {
      return specfem::display::wavefield::acceleration;
    } else {
      throw std::runtime_error("Unknown wavefield type");
    }
  }();

  return std::make_shared<specfem::writer::plot_wavefield>(
      assembly, output_format, wavefield_type, this->output_folder);
}
