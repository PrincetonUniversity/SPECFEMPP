#ifndef _COMPUTE_QUADRATURE_HPP
#define _COMPUTE_QUADRATURE_HPP

#include "quadrature/interface.hpp"

namespace specfem {
namespace compute {

struct quadrature {
  specfem::quadrature::gll::gll gll;

  quadrature(const specfem::quadrature::gll::gll &gll) : gll(gll) {}
};

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_QUADRATURE_HPP */
