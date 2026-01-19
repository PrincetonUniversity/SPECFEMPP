#include "header.hpp"
#include "yaml-cpp/yaml.h"
#include <ostream>

specfem::runtime_configuration::header::header(const YAML::Node &Node) {

  try {
    *this = specfem::runtime_configuration::header(
        Node["title"].as<std::string>(), Node["description"].as<std::string>());
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading header config. \n" << e.what();

    throw std::runtime_error(message.str());
  }
}
