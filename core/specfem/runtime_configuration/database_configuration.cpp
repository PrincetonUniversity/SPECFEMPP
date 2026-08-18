#include "database_configuration.hpp"

#include "yaml-cpp/yaml.h"
#include <ostream>

specfem::runtime_configuration::database_configuration::database_configuration(
    const YAML::Node &database_node) {
  try {
    *this = specfem::runtime_configuration::database_configuration(
        database_node["mesh-database"].as<std::string>());

  } catch (YAML::ParserException &e) {

    std::ostringstream message;

    message << "Error reading database configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }
}
