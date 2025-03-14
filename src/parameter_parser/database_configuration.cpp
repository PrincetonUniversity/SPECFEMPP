#include "parameter_parser/database_configuration.hpp"
#include "specfem_mpi/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <ostream>

specfem::runtime_configuration::database_configuration::database_configuration(
    const YAML::Node &database_node) {
  try {
    if (database_node.size() == 1) {
      *this = specfem::runtime_configuration::database_configuration(
          database_node["mesh-database"].as<std::string>());
    } else if (database_node.size() == 2) {
      *this = specfem::runtime_configuration::database_configuration(
          database_node["mesh-database"].as<std::string>(),
          database_node["mesh-parameters"].as<std::string>());

    } else {
      throw std::runtime_error(
          "Error reading database configuration. Node size "
          "is not 1 or 2");
    }

  } catch (YAML::ParserException &e) {

    std::ostringstream message;

    message << "Error reading database configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }
}
