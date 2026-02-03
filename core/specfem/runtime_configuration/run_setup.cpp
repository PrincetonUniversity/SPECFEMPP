#include "run_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <ostream>

specfem::runtime_configuration::run_setup::run_setup(const YAML::Node &Node) {
  try {
    *this = specfem::runtime_configuration::run_setup(
        Node["number-of-processors"].as<int>(),
        Node["number-of-runs"].as<int>());
  } catch (YAML::ParserException &e) {
    std::ostringstream message;
    message << "Error reading run setup. " << e.what();

    throw std::runtime_error(message.str());
  }
}
