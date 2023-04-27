#include "parameter_parser/interface.hpp"
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
      this->stypes.push_back(specfem::seismogram::acceleration);
    } else {
      std::runtime_error("Seismograms config could not be read properly");
    }
  }

  return;
}
