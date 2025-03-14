#include "parameter_parser/elastic_wave.hpp"

specfem::runtime_configuration::elastic_wave::elastic_wave(
    const YAML::Node &Node) {
  try {
    this->elastic_wave_type = Node.as<std::string>();
  } catch (YAML::Exception &e) {
    throw std::runtime_error("Error reading elastic wave type: " +
                             std::string(e.what()));
  }

  if (this->elastic_wave_type != "P_SV" && this->elastic_wave_type != "SH") {
    throw std::runtime_error("Invalid elastic wave type: " +
                             this->elastic_wave_type);
  }

  return;
}
