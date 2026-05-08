#include "flux_schemes.hpp"
#include "specfem/element_coupling.hpp"
#include <stdexcept>
#include <utility>
#include <yaml-cpp/exceptions.h>

specfem::runtime_configuration::flux_schemes::flux_schemes(
    const YAML::Node &Node)
    : flux_schemes() // inherit default values from default constructor
{
  flux_schemes_node = Node;

  // multi-rules in commit 4a27c96cde7e8548556da6caf628eda034fd35b3

  if (const YAML::Node flux_scheme_node = Node["type"]) {
    try {
      std::string flux_scheme_str = flux_scheme_node.as<std::string>();
      if (flux_scheme_str == "natural") {
        flux_scheme_tag = specfem::element_coupling::flux_scheme_tag::natural;
      } else {
        throw std::runtime_error("Unsupported flux scheme type: " +
                                 flux_scheme_str);
      }
    } catch (const YAML::Exception &e) {
      throw std::runtime_error("Error parsing flux scheme configuration: " +
                               std::string(e.what()));
    }
  }

  if (const YAML::Node quadrature_node = Node["quadrature"]) {

    try { // generate interfacial_quadrature by
          // runtime_configuration::quadrature
      interfacial_quadrature =
          std::make_unique<specfem::runtime_configuration::quadrature>(
              quadrature_node);
    } catch (const YAML::Exception &e) {
      throw std::runtime_error("Error parsing coupling quadrature rule: " +
                               std::string(e.what()));
    }
  }

  if (const YAML::Node interfacial_meshing_type_node =
          Node["interfacial-mesh"]) {
    try {
      std::string meshing_type_str =
          interfacial_meshing_type_node.as<std::string>();
      // consider replacing with from_string() or add alias handling.
      if (meshing_type_str == "intersections") {
        interfacial_meshing_type =
            specfem::element_coupling::interfacial_meshing_type::intersections;
      } else if (meshing_type_str == "acoustic-host") {
        interfacial_meshing_type =
            specfem::element_coupling::interfacial_meshing_type::acoustic_host;
      } else if (meshing_type_str == "elastic-host") {
        interfacial_meshing_type =
            specfem::element_coupling::interfacial_meshing_type::elastic_host;
      } else if (meshing_type_str == "self-host") {
        interfacial_meshing_type =
            specfem::element_coupling::interfacial_meshing_type::self_host;
      } else {
        throw std::runtime_error("Unsupported interfacial meshing type: " +
                                 meshing_type_str);
      }
    } catch (const YAML::Exception &e) {
      throw std::runtime_error("Error parsing interfacial_meshing_type: " +
                               std::string(e.what()));
    }
  }
}

specfem::element_coupling::flux_scheme_configuration
specfem::runtime_configuration::flux_schemes::get_flux_scheme(
    const int &ngll) const {
  specfem::element_coupling::flux_scheme_configuration config;

  config.set_flux_scheme_tag(flux_scheme_tag);
  config.set_interfacial_meshing_type(interfacial_meshing_type);

  if (interfacial_quadrature != nullptr) {
    config.set_interfacial_quadrature(
        interfacial_quadrature->instantiate().gll);
  } else {
    // default interfacial quadrature rule (GLL-N)
    // TODO: replace with GL-(N-1) for interfacial_meshing_type::intersections,
    // (GLL-N) for host
    config.set_interfacial_quadrature(
        specfem::quadrature::gll::gll(0.0, 0.0, ngll));
  }

  return config;
}
