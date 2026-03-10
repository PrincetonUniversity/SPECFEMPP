#pragma once

#include "specfem/constants.hpp"
#include "specfem/element_coupling.hpp"
namespace specfem::element_coupling {

struct flux_scheme_configuration {
  specfem::element_coupling::flux_scheme_tag flux_scheme_tag;
  std::vector<type_real> scheme_parameters;

  flux_scheme_configuration()
      : flux_scheme_tag(specfem::element_coupling::flux_scheme_tag::natural) {}

  /**
   * @brief Get the flux scheme tag for a given intersection between two
   * materials.
   *
   * @param interface_tag - interface_tag of the intersection; output is same
   * between a given tag and its conjugate.
   * @param material1 - one material index. Can be swapped with material2.
   * @param material2 - the other material index. Can be swapped with material1.
   * @return specfem::element_coupling::flux_scheme_tag the flux scheme
   * corresponding to the intersection.
   */
  specfem::element_coupling::flux_scheme_tag get_flux_scheme_tag(
      const specfem::element_coupling::interface_tag &interface_tag,
      const int &material1, const int &material2) const {
    return flux_scheme_tag;
  }
};
} // namespace specfem::element_coupling
