#ifndef _POINT_COORDINATES_HPP
#define _POINT_COORDINATES_HPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

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

struct index {
  int ispec;
  int iz;
  int ix;

  index() = default;

  index(const int &ispec, const int &iz, const int &ix)
      : ispec(ispec), iz(iz), ix(ix) {}
};

type_real distance(const specfem::point::gcoord2 &p1,
                   const specfem::point::gcoord2 &p2);

} // namespace point
} // namespace specfem

#endif
