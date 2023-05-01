#include "parameter_parser/interface.hpp"
#include "yaml-cpp/yaml.h"

specfem::runtime_configuration::database_configuration::database_configuration(
    const YAML::Node &Node) {
  *this = specfem::runtime_configuration::database_configuration(
      Node["mesh-database"].as<std::string>(),
      Node["source-file"].as<std::string>());
}
