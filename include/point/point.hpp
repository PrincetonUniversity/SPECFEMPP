#ifndef _SPECFEM_POINT_HPP
#define _SPECFEM_POINT_HPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem {
namespace point {

template <int N> struct local {
  int ispec;
  specfem::kokkos::array_type<type_real, N> coordinates;

  local() = default;

  local(const int &ispec, const type_real &xi, const type_real &gamma)
      : ispec(ispec), coordinates({ xi, gamma }) {
    static_assert(N == 2, "Calling 2D constructor for 3D point");
  }

  local(const int &ispec, const type_real &xi, const type_real &gamma,
        const type_real &eta)
      : ispec(ispec), coordinates({ xi, eta, gamma }) {
    static_assert(N == 3, "Calling 3D constructor for 2D point");
  }
};

template <int N> struct global {
  specfem::kokkos::array_type<type_real, N> coordinates;

  global() = default;

  global(const type_real &x, const type_real &z) : coordinates({ x, z }) {
    static_assert(N == 2, "Calling 2D constructor for 3D point");
  }

  global(const type_real &x, const type_real &z, const type_real &y)
      : coordinates({ x, y, z }) {
    static_assert(N == 3, "Calling 3D constructor for 2D point");
  }
};

template <int N>
type_real distance(const specfem::point::global<N> &p1,
                   const specfem::point::global<N> &p2) {
  type_real dist = 0;
  for (int i = 0; i < N; i++) {
    dist += (p1.coordinates[i] - p2.coordinates[i]) *
            (p1.coordinates[i] - p2.coordinates[i]);
  }
  return std::sqrt(dist);
}

} // namespace point
} // namespace specfem

#endif
