#pragma once

#include "specfem/constants.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"

#include "specfem/runtime_configuration/quadrature.hpp"
#include "yaml-cpp/yaml.h"

#include <memory>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief class to read flux-scheme information
 *
 */
class flux_schemes {
public:
  flux_schemes(const YAML::Node &Node);
  flux_schemes() // default values here; noded constructor delegates to this
                 // one:
                 //
                 // (TODO: remember to re-examine interfacial_meshing_type
                 // after self-host study)
      : flux_scheme_tag(specfem::element_coupling::flux_scheme_tag::natural),
        interfacial_quadrature(nullptr),
        interfacial_meshing_type(specfem::element_coupling::
                                     interfacial_meshing_type::intersections) {
        };

private:
  // ====== acoustic-elastic ======
  specfem::element_coupling::flux_scheme_tag flux_scheme_tag;
  specfem::element_coupling::interfacial_meshing_type interfacial_meshing_type;
  std::unique_ptr<specfem::runtime_configuration::quadrature>
      interfacial_quadrature;
  // ==== end acoustic-elastic ====

public:
  /**
   * @brief Generates a flux_scheme_configuration object from the given runtime
   * configuration.
   *
   * @param ngll number of element quadrature points of the simulation. This is
   * used for setting the default value of the interfacial quadrature rule.
   * @return specfem::element_coupling::flux_scheme_configuration the generated
   * configuration to be given to assembly.
   */
  specfem::element_coupling::flux_scheme_configuration
  get_flux_scheme(const int &ngll) const;

  YAML::Node flux_schemes_node;
};
} // namespace runtime_configuration
} // namespace specfem
