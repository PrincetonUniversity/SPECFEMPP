#ifndef _QUADRATURE_QUADRATURES_HPP
#define _QUADRATURE_QUADRATURES_HPP

#include "quadrature/gll/interface.hpp"

namespace specfem {
namespace quadrature {
struct quadratures {
  specfem::quadrature::gll::gll gll;

  quadratures(const specfem::quadrature::gll::gll &gll) : gll(gll) {}
};
} // namespace quadrature
} // namespace specfem

#endif
