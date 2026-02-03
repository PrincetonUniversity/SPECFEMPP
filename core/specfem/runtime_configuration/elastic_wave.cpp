#include "elastic_wave.hpp"
#include "specfem/utilities.hpp"

specfem::runtime_configuration::elastic_wave::elastic_wave(
    const YAML::Node &Node) {
  try {
    this->elastic_wave_type = Node.as<std::string>();
  } catch (YAML::Exception &e) {
    throw std::runtime_error("Error reading elastic wave type: " +
                             std::string(e.what()));
  }

  if (!specfem::utilities::is_psv_string(this->elastic_wave_type) &&
      !specfem::utilities::is_sh_string(this->elastic_wave_type)) {
    throw std::runtime_error("Invalid elastic wave type: " +
                             this->elastic_wave_type);
  }

  return;
}
