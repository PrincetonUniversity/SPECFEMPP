#include "parameter_parser/interface.hpp"
#include "yaml-cpp/yaml.h"

specfem::runtime_configuration::header::header(const YAML::Node &Node) {
  *this = specfem::runtime_configuration::header(
      Node["title"].as<std::string>(), Node["description"].as<std::string>());
}
