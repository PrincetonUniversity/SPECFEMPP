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

  for (const YAML::Node entry : this->flux_schemes_node) {

    specfem::element_coupling::flux_scheme_tag flux_scheme_tag;

    // for now, we will only assume this is a singleton with "interface"
    // identifier and not "material<1/2>". This code should work if generalized,
    // but has not been verified.
    if (const YAML::Node &flux_scheme_type_node = entry["type"]) {
      try {
        const auto flux_scheme_type = flux_scheme_type_node.as<std::string>();

        // TODO flux_scheme_tag from_string() function?
        if (flux_scheme_type == std::string("natural")) {
          flux_scheme_tag = specfem::element_coupling::flux_scheme_tag::natural;
        } else {
          throw std::runtime_error(std::string("Unrecognized flux scheme: \"") +
                                   flux_scheme_type + "\"");
        }

        // two cases: interface specification or material specification -- only
        // one can be set for a given rule
        const YAML::Node &interface_identifier_node = entry["interface"];
        const YAML::Node &material1_identifier_node = entry["material1"];
        const YAML::Node &material2_identifier_node = entry["material2"];
        bool is_interface_defined = interface_identifier_node.IsDefined();
        bool is_material_defined = material1_identifier_node.IsDefined() ||
                                   material2_identifier_node.IsDefined();

        if (is_interface_defined) {
          if (is_material_defined) {
            throw std::runtime_error(
                "Flux-scheme entry has both \"interface\"-specification and "
                "\"material1/2\"-specification. Only one is permitted.");
          }
          try {
            const auto interface_identifier =
                interface_identifier_node.as<std::string>();

            // TODO interface_tag from_string() function?
            specfem::element_coupling::interface_tag interface_tag;
            if (interface_identifier == std::string("acoustic_elastic") ||
                interface_identifier == std::string("elastic_acoustic")) {
              interface_tag =
                  specfem::element_coupling::interface_tag::acoustic_elastic;
            } else {
              throw std::runtime_error(
                  std::string("Unrecognized interface tag: \"") +
                  interface_identifier + "\"");
            }

            // append entry
            this->per_interface_configs[interface_tag] =
                std::make_pair(flux_scheme_tag, entry);

          } catch (YAML::InvalidNode &e) {
            throw std::runtime_error("Flux-scheme entry has \"type\" node, "
                                     "but it is not a string!");
          }
          continue;
        }

        if (is_material_defined) {
          if ((!material1_identifier_node.IsDefined()) ||
              (!material2_identifier_node.IsDefined())) {
            throw std::runtime_error(
                "When a flux-scheme entry uses a "
                "\"material1/2\"-specification, both \"material1\" and "
                "\"material2\" must be specified!");
          }
          try {
            int material1 = material1_identifier_node.as<int>();
            int material2 = material2_identifier_node.as<int>();

            // use lowest number first:
            if (material1 > material2) {
              std::swap(material1, material2);
            }

            // append entry
            this->per_material_configs[std::make_pair(material1, material2)] =
                std::make_pair(flux_scheme_tag, entry);

          } catch (YAML::InvalidNode &e) {
            throw std::runtime_error("Flux-scheme entry has \"type\" node, "
                                     "but it is not a string!");
          }
          continue;
        }

        // no specification made
        throw std::runtime_error(
            "Flux-scheme needs either \"interface\"-specification or "
            "\"material1/2\"-specification.");
      } catch (YAML::InvalidNode &e) {
        throw std::runtime_error(
            "Flux-scheme entry has \"type\" node, but it is not a string!");
      }
    } else {
      throw std::runtime_error(
          "Flux-scheme entry must have a \"type\" string specified.");
    }
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
