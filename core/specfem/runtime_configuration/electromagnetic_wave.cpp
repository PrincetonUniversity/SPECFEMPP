#include "electromagnetic_wave.hpp"

specfem::runtime_configuration::electromagnetic_wave::electromagnetic_wave(
    const YAML::Node &Node) {
  try {
    this->electromagnetic_wave_type = Node.as<std::string>();
  } catch (YAML::Exception &e) {
    throw std::runtime_error("Error reading electromagnetic wave type: " +
                             std::string(e.what()));
  }

  if (this->electromagnetic_wave_type != "TE" &&
      this->electromagnetic_wave_type != "TM") {
    throw std::runtime_error("Invalid electromagnetic wave type: " +
                             this->electromagnetic_wave_type);
  }

  return;
}
