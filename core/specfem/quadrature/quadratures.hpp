#pragma once

#include "gll.hpp"

namespace specfem {
namespace quadrature {
struct quadratures {
  specfem::quadrature::gll::gll gll;

  quadratures(const specfem::quadrature::gll::gll &gll) : gll(gll) {}

  /**
   * @brief Get string representation of the quadratures
   */
  std::string to_string() const {
    std::ostringstream out;
    out << "GLL Quadratures:\n";
    out << gll.to_string() << "\n";
    return out.str();
  }
};
} // namespace quadrature
} // namespace specfem
