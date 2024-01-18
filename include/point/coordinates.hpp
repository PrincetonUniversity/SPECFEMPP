#ifndef _SPECFEM_POINT_HPP
#define _SPECFEM_POINT_HPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem {
namespace point {

struct lcoord2 {
  int ispec;
  type_real xi;
  type_real gamma;

  lcoord2() = default;

  lcoord2(const int &ispec, const type_real &xi, const type_real &gamma)
      : ispec(ispec), xi(xi), gamma(gamma) {}
};

struct gcoord2 {
  type_real x;
  type_real z;

  gcoord2() = default;

  gcoord2(const type_real &x, const type_real &z) : x(x), z(z) {}
};

template <int N>
type_real distance(const specfem::point::gcoord2 &p1,
                   const specfem::point::gcoord2 &p2) {
  return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                   (p1.z - p2.z) * (p1.z - p2.z));
}

} // namespace point
} // namespace specfem

#endif
