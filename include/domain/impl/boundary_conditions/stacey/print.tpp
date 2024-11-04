#pragma once

#include "domain/impl/boundary_conditions/boundary_conditions.hpp"

template <>
std::string
specfem::domain::impl::boundary_conditions::print_boundary_tag<
    specfem::element::boundary_tag::stacey>() {
  return "stacey";
}
