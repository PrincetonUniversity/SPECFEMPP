#ifndef _ALGORITHMS_INTERPOLATE_HPP
#define _ALGORITHMS_INTERPOLATE_HPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION T interpolate_function(
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
KOKKOS_FUNCTION T interpolate_function(
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

template <int components, typename Layout, typename MemorySpace,
          typename MemoryTraits>
KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, components>
interpolate_function(
    const Kokkos::View<type_real **, specfem::kokkos::LayoutWrapper,
                       MemorySpace, MemoryTraits> &polynomial,
    const Kokkos::View<type_real **[components], Layout, MemorySpace,
                       MemoryTraits> &function) {

  using T = specfem::kokkos::array_type<type_real, components>;

  const int N = polynomial.extent(0);
  T result(0.0);

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<MemorySpace, Kokkos::Rank<2> >({ 0, 0 }, { N, N }),
      [&](const int iz, const int ix, T &sum) {
        for (int icomponent = 0; icomponent < components; ++icomponent) {
          sum[icomponent] += polynomial(iz, ix) * function(iz, ix, icomponent);
        }
      },
      specfem::kokkos::Sum<T>(result));

  return result;
}

template <int components, typename Layout, typename MemorySpace,
          typename MemoryTraits>
KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, components>
interpolate_function(
    const typename Kokkos::TeamPolicy<MemorySpace>::member_type &team_member,
    const Kokkos::View<type_real **, specfem::kokkos::LayoutWrapper,
                       MemorySpace, MemoryTraits> &polynomial,
    const Kokkos::View<type_real **[components], Layout, MemorySpace,
                       MemoryTraits> &function) {

  using T = specfem::kokkos::array_type<type_real, components>;

  const int N = polynomial.extent(0);
  T result(0.0);

  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, N * N),
      [&](const int &xz, T &sum) {
        int iz, ix;
        sub2ind(xz, N, iz, ix);
        for (int icomponent = 0; icomponent < components; ++icomponent) {
          sum[icomponent] += polynomial(iz, ix) * function(iz, ix, icomponent);
        }
      },
      specfem::kokkos::Sum<T>(result));

  return result;
}

} // namespace algorithms
} // namespace specfem

#endif
