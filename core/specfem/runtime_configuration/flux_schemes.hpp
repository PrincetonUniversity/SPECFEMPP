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
  flux_schemes()
      : flux_scheme_tag(specfem::element_coupling::flux_scheme_tag::natural) {};

private:
  specfem::element_coupling::flux_scheme_tag flux_scheme_tag;

public:
  specfem::element_coupling::flux_scheme_configuration get_flux_scheme() const {
    specfem::element_coupling::flux_scheme_configuration config;

    config.flux_scheme_tag = flux_scheme_tag;

    return config;
  }

  YAML::Node flux_schemes_node;
};
} // namespace runtime_configuration
} // namespace specfem
