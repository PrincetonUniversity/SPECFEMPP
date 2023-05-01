#include "parameter_parser/interface.hpp"
#include "yaml-cpp/yaml.h"

specfem::runtime_configuration::run_setup::run_setup(const YAML::Node &Node) {
  *this = specfem::runtime_configuration::run_setup(
      Node["number-of-processors"].as<int>(), Node["number-of-runs"].as<int>());
}
