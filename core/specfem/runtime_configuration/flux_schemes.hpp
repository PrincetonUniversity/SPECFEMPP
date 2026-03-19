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

private:
  bool config_provided;
  YAML::Node flux_schemes_node; /// Node that contains flux scheme information

  std::map<specfem::element_coupling::interface_tag,
           std::pair<specfem::element_coupling::flux_scheme_tag, YAML::Node> >
      per_interface_configs;

  std::map<std::pair<int, int>,
           std::pair<specfem::element_coupling::flux_scheme_tag, YAML::Node> >
      per_material_configs;

public:
  specfem::element_coupling::flux_scheme_configuration generate_configuration();
};
} // namespace runtime_configuration
} // namespace specfem
