#include "parameter_parser/database_configuration.hpp"
#include "yaml-cpp/yaml.h"
#include <ostream>

specfem::runtime_configuration::database_configuration::database_configuration(
    const YAML::Node &Node) {
  try {
    if (const YAML::Node &source_node = Node["sources"]) {
      *this = specfem::runtime_configuration::database_configuration(
          Node["mesh-database"].as<std::string>(),
          source_node.as<std::string>());
    } else if (const YAML::Node &source_node = Node["source-dict"]) {
      *this = specfem::runtime_configuration::database_configuration(
          Node["mesh-database"].as<std::string>(), source_node);
    } else {
      throw std::runtime_error("Error reading database configuration source "
                               "field not recognized. \n");
    }

  } catch (YAML::ParserException &e) {

    std::ostringstream message;

    message << "Error reading database configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }
}
