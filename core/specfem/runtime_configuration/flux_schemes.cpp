#include "flux_schemes.hpp"
#include "specfem/element_coupling.hpp"
#include <stdexcept>
#include <utility>
#include <yaml-cpp/exceptions.h>

specfem::runtime_configuration::flux_schemes::flux_schemes(
    const YAML::Node &Node)
    : flux_schemes_node(Node) {

  // multi-rules in commit 4a27c96cde7e8548556da6caf628eda034fd35b3

  try {
    std::string flux_scheme_str = Node["type"].as<std::string>();
    if (flux_scheme_str == "natural") {
      flux_scheme_tag = specfem::element_coupling::flux_scheme_tag::natural;
    } else {
      throw std::runtime_error("Unsupported flux scheme type: " +
                               Node["type"].as<std::string>());
    }
  } catch (const YAML::Exception &e) {
    throw std::runtime_error("Error parsing flux scheme configuration: " +
                             std::string(e.what()));
  }
}
