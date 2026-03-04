#include "flux_schemes.hpp"
#include "specfem/element_coupling.hpp"
#include <stdexcept>
#include <utility>
#include <yaml-cpp/exceptions.h>

specfem::runtime_configuration::flux_schemes::flux_schemes(
    const YAML::Node &Node)
    : flux_schemes_node(Node) {
  if (!this->flux_schemes_node.IsSequence()) {
    throw std::runtime_error("flux-scheme YAML node must be a sequence.");
  }

  // multi-rules in commit 4a27c96cde7e8548556da6caf628eda034fd35b3

  if (const YAML::Node &flux_scheme_type_node = flux_schemes_node["type"]) {
    specfem::element_coupling::flux_scheme_tag flux_scheme_tag;
    std::string flux_scheme_str;
    try {
      flux_scheme_str = flux_scheme_type_node.as<std::string>();
    } catch (YAML::InvalidNode &e) {
      throw std::runtime_error(
          "Flux-scheme entry has \"type\" node, but it is not a string!");
    }

    // === parse flux scheme string to tag and verify necessary parameters ===

    // TODO flux_scheme_tag from_string() function?
    if (flux_scheme_str == std::string("natural")) {
      flux_scheme_tag = specfem::element_coupling::flux_scheme_tag::natural;
    } else if (flux_scheme_str == std::string("symmetric_interior_penalty")) {
      flux_scheme_tag = specfem::element_coupling::flux_scheme_tag::
          symmetric_interior_penalty;
      throw std::runtime_error(
          "symmetric_interior_penalty not yet supported. Do not set this "
          "scheme in your specfem configuration.");
    } else {
      throw std::runtime_error(std::string("Unrecognized flux scheme: \"") +
                               flux_scheme_str + "\"");
    }

    // === set config ===
    per_interface_configs
        [specfem::element_coupling::interface_tag::acoustic_elastic] =
            std::make_pair(flux_scheme_tag, flux_schemes_node);

  } else {
    throw std::runtime_error(
        "Flux-scheme entry must have a \"type\" string specified.");
  }
}

specfem::element_coupling::flux_scheme_configuration
specfem::runtime_configuration::flux_schemes::generate_configuration() {
  specfem::element_coupling::flux_scheme_configuration config;

  // for now, we will only assume this is a singleton with "interface"
  // identifier and not "material<1/2>". runtime_configuration::flux_schemes
  // should work with multiple, but element_coupling::flux_scheme_configuration
  // has not been implemented, so catch multiple-case here here.

  if (per_material_configs.size() > 0) {
    throw std::runtime_error(
        "Per-material flux-scheme configuration not implemented.");
  }
  if (per_interface_configs.size() > 1) {
    throw std::runtime_error(
        "Per-interface flux-scheme configuration only supports 1 type.");
  }

  if (per_interface_configs.size() == 1) {
    const auto config_entry = per_interface_configs.begin()->second;
    config.flux_scheme_tag = config_entry.first;
  }
  return config;
}
