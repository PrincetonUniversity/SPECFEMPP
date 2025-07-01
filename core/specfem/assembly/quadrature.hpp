#pragma once

#include "quadrature/interface.hpp"

namespace specfem::assembly {

struct quadrature {
  specfem::quadrature::gll::gll gll;

  quadrature(const specfem::quadrature::gll::gll &gll) : gll(gll) {}
};

} // namespace specfem::assembly
