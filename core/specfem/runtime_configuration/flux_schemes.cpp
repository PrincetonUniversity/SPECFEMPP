#include "flux_schemes.hpp"
#include "specfem/element_coupling.hpp"
#include <stdexcept>
#include <utility>
#include <yaml-cpp/exceptions.h>

specfem::runtime_configuration::flux_schemes::flux_schemes(
    const YAML::Node &Node)
    : flux_schemes_node(Node) {

  try {
    if (Node["type"].as<std::string>() == "natural") {
      flux_scheme = "natural";
      flux_scheme_configuration = std::make_shared<configuration_details<
          specfem::element_coupling::flux_scheme_tag::natural> >(Node);
    } else {
      throw std::runtime_error("Unsupported flux scheme type: " +
                               Node["type"].as<std::string>());
    }
  } catch (const YAML::Exception &e) {
    throw std::runtime_error("Error parsing flux scheme configuration: " +
                             std::string(e.what()));
  }
}
