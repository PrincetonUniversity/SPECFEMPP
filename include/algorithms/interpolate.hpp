#ifndef _ALGORITHMS_INTERPOLATE_HPP
#define _ALGORITHMS_INTERPOLATE_HPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

template <typename T, typename MemorySpace>
T interpolate_function(
    const Kokkos::View<type_real **, specfem::kokkos::LayoutWrapper,
                       MemorySpace> &polynomial,
    const Kokkos::View<T **, specfem::kokkos::LayoutWrapper, MemorySpace>
        &function) {

  using ExecSpace = typename MemorySpace::execution_space;

  const int N = polynomial.extent(0);
  T result(0.0);

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2> >({ 0, 0 }, { N, N }),
      KOKKOS_LAMBDA(const int iz, const int ix, T &sum) {
        sum += polynomial(iz, ix) * function(iz, ix);
      },
      result);

  return result;
}

template <typename T, typename ExecSpace>
T interpolate_function(
    const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team_member,
    const Kokkos::View<type_real **, specfem::kokkos::LayoutWrapper,
                       typename ExecSpace::memory_space> &polynomial,
    const Kokkos::View<T **, specfem::kokkos::LayoutWrapper,
                       typename ExecSpace::memory_space> &function) {

  const int N = polynomial.extent(0);

  T result(0.0);
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, N * N),
      [&](const int &xz, T &sum) {
        int iz, ix;
        sub2ind(xz, N, iz, ix);
        sum += polynomial(iz, ix) * function(iz, ix);
      },
      result);

  return result;
}

} // namespace algorithms
} // namespace specfem

#endif
