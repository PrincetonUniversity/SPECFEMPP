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

  KOKKOS_FUNCTION
  lcoord2() = default;

  KOKKOS_FUNCTION
  lcoord2(const int &ispec, const type_real &xi, const type_real &gamma)
      : ispec(ispec), xi(xi), gamma(gamma) {}
};

struct gcoord2 {
  type_real x;
  type_real z;

  KOKKOS_FUNCTION
  gcoord2() = default;

  KOKKOS_FUNCTION
  gcoord2(const type_real &x, const type_real &z) : x(x), z(z) {}
};

struct index {
  int ispec;
  int iz;
  int ix;

  KOKKOS_FUNCTION
  index() = default;

  KOKKOS_FUNCTION
  index(const int &ispec, const int &iz, const int &ix)
      : ispec(ispec), iz(iz), ix(ix) {}
};

struct simd_index {
  int ispec;
  int number_elements;
  int iz;
  int ix;

  KOKKOS_FUNCTION
  bool mask(const std::size_t &lane) const {
    return int(lane) < number_elements;
  }

  KOKKOS_FUNCTION
  simd_index() = default;

  KOKKOS_FUNCTION
  simd_index(const int &ispec, const int &number_elements, const int &iz,
             const int &ix)
      : ispec(ispec), number_elements(number_elements), iz(iz), ix(ix) {}
};

KOKKOS_FUNCTION
type_real distance(const specfem::point::gcoord2 &p1,
                   const specfem::point::gcoord2 &p2);

} // namespace point
} // namespace specfem

#endif
