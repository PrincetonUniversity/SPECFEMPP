#pragma once

#include "quadrature/interface.hpp"

namespace specfem {
namespace compute {

struct quadrature {
  specfem::quadrature::gll::gll gll;

  quadrature(const specfem::quadrature::gll::gll &gll) : gll(gll) {}
};

} // namespace compute
} // namespace specfem
