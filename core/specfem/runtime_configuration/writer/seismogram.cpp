#include "seismogram.hpp"
#include "specfem/io.hpp"
#include "specfem/utilities.hpp"
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

  const std::optional<bool> write_from_main = [&]() -> std::optional<bool> {
    if (seismogram["write-from-main"]) {
      return seismogram["write-from-main"].as<bool>();
    } else {
      return std::nullopt;
    }
  }();

  if (!boost::filesystem::is_directory(
          boost::filesystem::path(output_folder))) {
    std::ostringstream message;
    message << "Output folder : " << output_folder << " does not exist.";
    throw std::runtime_error(message.str());
  }

  *this = specfem::runtime_configuration::seismogram(
      output_format, output_folder, write_from_main);

  return;
}

std::shared_ptr<specfem::io::writer>
specfem::runtime_configuration::seismogram::instantiate_seismogram_writer(
    const specfem::enums::elastic_wave elastic_wave,
    const specfem::enums::electromagnetic_wave electromagnetic_wave,
    const type_real dt, const type_real t0, const int nstep_between_samples,
    std::optional<specfem::datetime::type> starttime) const {

  const auto type = [&]() {
    if (specfem::utilities::is_su_string(this->output_format)) {
      throw std::runtime_error("Seismic Unix format not implemented yet");
      return specfem::enums::seismogram_format::seismic_unix;
    } else if (specfem::utilities::is_sac_string(this->output_format)) {
      return specfem::enums::seismogram_format::sac;
    } else if (specfem::utilities::is_ascii_string(this->output_format)) {
      return specfem::enums::seismogram_format::ascii;
    } else {
      throw std::runtime_error("Unknown seismogram format");
    }
  }();

  std::shared_ptr<specfem::io::writer> writer =
      std::make_shared<specfem::io::seismogram_writer>(
          type, elastic_wave, electromagnetic_wave, this->output_folder, dt, t0,
          nstep_between_samples, starttime, this->write_from_main);

  return writer;
}
