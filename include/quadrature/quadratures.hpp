#ifndef _QUADRATURE_QUADRATURES_HPP
#define _QUADRATURE_QUADRATURES_HPP

#include "quadrature/gll/interface.hpp"

namespace specfem {
namespace quadrature {
struct quadratures {
  specfem::quadrature::gll::gll gll;

  quadratures(const specfem::quadrature::gll::gll &gll) : gll(gll) {}

  void print(std::ostream &out) const {
    out << " Quadrature in X-dimension \n";
    gll.print(out);

    out << " Quadrature in Z-dimension \n";
    gll.print(out);
  }
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::quadrature::quadratures &quad);
} // namespace quadrature
} // namespace specfem

#endif
