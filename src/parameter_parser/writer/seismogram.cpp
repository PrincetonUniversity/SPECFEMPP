#include "constants.hpp"
#include "parameter_parser/interface.hpp"
#include "writer/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <boost/filesystem.hpp>
#include <string>

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

  try {
    *this = specfem::runtime_configuration::seismogram(
        seismogram["output-format"].as<std::string>(), output_folder);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading seismogram config. \n" << e.what();

    std::runtime_error(message.str());
  }

  return;
}

std::shared_ptr<specfem::writer::writer>
specfem::runtime_configuration::seismogram::instantiate_seismogram_writer(
    const specfem::compute::receivers &receivers, const type_real dt,
    const type_real t0, const int nstep_between_samples) const {

  const auto type = [&]() {
    if (this->output_format == "seismic_unix" || this->output_format == "su") {
      return specfem::enums::seismogram::format::seismic_unix;
    } else if (this->output_format == "ascii") {
      return specfem::enums::seismogram::format::ascii;
    } else {
      throw std::runtime_error("Unknown seismogram format");
    }
  }();

  std::shared_ptr<specfem::writer::writer> writer =
      std::make_shared<specfem::writer::seismogram>(
          receivers, type, this->output_folder, dt, t0, nstep_between_samples);

  return writer;
}
