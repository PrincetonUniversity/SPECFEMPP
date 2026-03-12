#pragma once

#include "specfem/constants.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"

#include "yaml-cpp/yaml.h"
#include <map>
#include <string>
#include <utility>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief class to read flux-scheme information
 *
 */
class flux_schemes {
public:
  flux_schemes(const YAML::Node &Node);

  std::shared_ptr<configuration> get_flux_scheme() const {
    return flux_scheme_configuration;
  }

private:
  struct configuration {
    specfem::element_coupling::flux_scheme_tag flux_scheme_tag;
  };

  template <specfem::element_coupling::flux_scheme_tag Tag>
  struct configuration_details;

  template <>
  struct configuration_details<
      specfem::element_coupling::flux_scheme_tag::natural>
      : public configuration {

    configuration_details(const YAML::Node &Node)
        : configuration{ specfem::element_coupling::flux_scheme_tag::natural } {
      // no parameters for natural flux scheme
    }
  };

  std::string flux_scheme;
  std::shared_ptr<configuration> flux_scheme_configuration;
};
} // namespace runtime_configuration
} // namespace specfem
