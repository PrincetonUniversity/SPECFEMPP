#include "parameter_parser/writer/seismogram.hpp"
#include "IO/seismogram/writer.hpp"
#include "yaml-cpp/yaml.h"
#include <boost/filesystem.hpp>
#include <string>

specfem::runtime_configuration::seismogram::seismogram(
    const YAML::Node &seismogram) {

  const std::string output_folder = [&]() {
    if (seismogram["directory"]) {
      return seismogram["directory"].as<std::string>();
    } else {
      return boost::filesystem::current_path().string();
    }
  }();

  const std::string output_format = [&]() -> std::string {
    if (seismogram["format"]) {
      return seismogram["format"].as<std::string>();
    } else {
      return "ASCII";
    }
  }();

  if (!boost::filesystem::is_directory(
          boost::filesystem::path(output_folder))) {
    std::ostringstream message;
    message << "Output folder : " << output_folder << " does not exist.";
    throw std::runtime_error(message.str());
  }

  *this =
      specfem::runtime_configuration::seismogram(output_format, output_folder);

  return;
}

std::shared_ptr<specfem::IO::writer>
specfem::runtime_configuration::seismogram::instantiate_seismogram_writer(
    const type_real dt, const type_real t0,
    const int nstep_between_samples) const {

  const auto type = [&]() {
    if (this->output_format == "seismic_unix" || this->output_format == "su") {
      throw std::runtime_error("Seismic Unix format not implemented yet");
      return specfem::enums::seismogram::format::seismic_unix;
    } else if (this->output_format == "ASCII" ||
               this->output_format == "ascii") {
      return specfem::enums::seismogram::format::ascii;
    } else {
      throw std::runtime_error("Unknown seismogram format");
    }
  }();

  std::shared_ptr<specfem::IO::writer> writer =
      std::make_shared<specfem::IO::seismogram_writer>(
          type, this->output_folder, dt, t0, nstep_between_samples);

  return writer;
}
