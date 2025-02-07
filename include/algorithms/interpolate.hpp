#ifndef _ALGORITHMS_INTERPOLATE_HPP
#define _ALGORITHMS_INTERPOLATE_HPP

#include "datatypes/point_view.hpp"
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
specfem::datatype::ScalarPointViewType<type_real, components, false>
interpolate_function(
    const Kokkos::View<type_real **, specfem::kokkos::LayoutWrapper,
                       MemorySpace, MemoryTraits> &polynomial,
    const Kokkos::View<type_real **[components], Layout, MemorySpace,
                       MemoryTraits> &function) {

  using T =
      specfem::datatype::ScalarPointViewType<type_real, components, false>;

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
KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, components,
                                                       false>
interpolate_function(
    const typename Kokkos::TeamPolicy<MemorySpace>::member_type &team_member,
    const Kokkos::View<type_real **, specfem::kokkos::LayoutWrapper,
                       MemorySpace, MemoryTraits> &polynomial,
    const Kokkos::View<type_real **[components], Layout, MemorySpace,
                       MemoryTraits> &function) {

  using T =
      specfem::datatype::ScalarPointViewType<type_real, components, false>;

  const int N = polynomial.extent(0);
  T result(0.0);

  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, N * N),
      [&](const int &xz, T &sum) {
        int iz, ix;
        sub2ind(xz, N, iz, ix);
        for (int icomponent = 0; icomponent < components; ++icomponent) {
          sum(icomponent) += polynomial(iz, ix) * function(iz, ix, icomponent);
        }
      },
      specfem::kokkos::Sum<T>(result));

  return result;
}

template <typename MemberType, typename IteratorType,
          typename PolynomialViewTye, typename FunctionViewType,
          typename ResultType>
KOKKOS_FUNCTION void interpolate_function(const MemberType &team_member,
                                          const IteratorType &iterator,
                                          const PolynomialViewTye &polynomial,
                                          const FunctionViewType &function,
                                          ResultType &result) {

  static_assert(PolynomialViewTye::rank() == 4, "Polynomial must be a 4D view");
  static_assert(FunctionViewType::rank() == 4, "Function must be a 4D view");

  static_assert(ResultType::rank() == 2, "Result must be 2D views");

#ifndef NDEBUG

  if (polynomial.extent(0) != function.extent(0) ||
      polynomial.extent(1) != function.extent(1)) {
    Kokkos::abort("Polynomial and function must have the same size");
  }

  if (polynomial.extent(0) != result.extent(0)) {
    Kokkos::abort("Polynomial and result must have the same size");
  }

  if (function.extent(3) != result.extent(1)) {
    Kokkos::abort(
        "Function and result must have the same number of components");
  }
#endif

  // // Initialize result
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, iterator.number_of_elements()),
      [&](const int &ielement) {
        result(ielement, 0) = 0.0;
        result(ielement, 1) = 0.0;
      });

  team_member.team_barrier();

  const int ncomponents = function.extent(3);

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, iterator.chunk_size()),
      [&](const int i) {
        const auto iterator_index = iterator(i);
        const auto index = iterator_index.index;

        for (int icomponent = 0; icomponent < ncomponents; ++icomponent) {
          type_real polynomial_value = polynomial(
              iterator_index.ielement, index.iz, index.ix, icomponent);
          type_real function_value =
              function(iterator_index.ielement, index.iz, index.ix, icomponent);
          Kokkos::atomic_add(&result(iterator_index.ielement, icomponent),
                             polynomial_value * function_value);
        }
      });
}

} // namespace algorithms
} // namespace specfem

#endif
