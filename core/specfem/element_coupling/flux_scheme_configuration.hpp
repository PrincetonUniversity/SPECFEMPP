#pragma once

#include "specfem/constants.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/quadrature/quadrature.hpp"

namespace specfem::element_coupling {

struct flux_scheme_configuration {
  specfem::element_coupling::flux_scheme_tag flux_scheme_tag;
  std::vector<type_real> scheme_parameters;

  specfem::quadrature::quadrature interfacial_quadrature;

  flux_scheme_configuration()
      : flux_scheme_tag(specfem::element_coupling::flux_scheme_tag::natural) {}

  /**
   * @brief Get the flux scheme tag for a given intersection. At the moment, the
   * flux scheme tag is global, so only one tag will ever be returned per
   * simulation.
   *
   * @return specfem::element_coupling::flux_scheme_tag the flux scheme
   * corresponding to the intersection.
   */
  specfem::element_coupling::flux_scheme_tag get_flux_scheme_tag() const {
    return flux_scheme_tag;
  }

  /**
   * @brief Get the quadrature object corresponding to interfacial integration.
   *
   * @return specfem::quadrature::quadrature the quadrature object for the
   * interface.
   */
  specfem::quadrature::quadrature get_interfacial_quadrature() const {
    return interfacial_quadrature;
  }
};
} // namespace specfem::element_coupling
