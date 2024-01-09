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

  try {
    *this = specfem::runtime_configuration::seismogram(
        seismogram["seismogram-format"].as<std::string>(), output_folder);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading seismogram config. \n" << e.what();

    std::runtime_error(message.str());
  }

  return;
}
