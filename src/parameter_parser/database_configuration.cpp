#include "parameter_parser/database_configuration.hpp"
#include "specfem_mpi/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <ostream>

specfem::runtime_configuration::database_configuration::database_configuration(
    const YAML::Node &database_node) {
  try {
    if (database_node["mesh-parameters"]) {
      *this = specfem::runtime_configuration::database_configuration(
          database_node["mesh-database"].as<std::string>(),
          database_node["mesh-parameters"].as<std::string>());

    } else {
      *this = specfem::runtime_configuration::database_configuration(
          database_node["mesh-database"].as<std::string>());
    }

  } catch (YAML::ParserException &e) {

    std::ostringstream message;

    message << "Error reading database configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }
}
