#pragma once

#include "specfem/constants.hpp"
#include "specfem/element_coupling.hpp"
namespace specfem::element_coupling {

struct flux_scheme_configuration {
  specfem::element_coupling::flux_scheme_tag flux_scheme_tag;
  std::vector<type_real> scheme_parameters;

  flux_scheme_configuration()
      : flux_scheme_tag(specfem::element_coupling::flux_scheme_tag::natural) {}
};
} // namespace specfem::element_coupling
