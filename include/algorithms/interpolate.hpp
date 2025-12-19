#pragma once

#include "datatypes/point_view.hpp"
#include "execution/for_each_level.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

namespace impl {
template <typename PolynomialViewType, typename FunctionViewType>
struct InterpolateFunctor {
  const PolynomialViewType polynomial;
  const FunctionViewType function;

  constexpr static int rank = PolynomialViewType::rank();

  InterpolateFunctor(const PolynomialViewType &polynomial,
                     const FunctionViewType &function)
      : polynomial(polynomial), function(function) {}

  template <typename T, int U = rank, std::enable_if_t<U == 2, int> = 0>
  KOKKOS_INLINE_FUNCTION void operator()(const int &iz, const int &ix,
                                         T &sum) const {
    sum += polynomial(iz, ix) * function(iz, ix);
  }

  template <typename T, int U = rank, std::enable_if_t<U == 3, int> = 0>
  KOKKOS_INLINE_FUNCTION void operator()(const int &iz, const int &iy,
                                         const int &ix, T &sum) const {
    sum += polynomial(iz, iy, ix) * function(iz, iy, ix);
  }
};
} // namespace impl

template <typename PolynomialViewType, typename FunctionViewType,
          std::enable_if_t<((PolynomialViewType::rank() == 2) &&
                            (FunctionViewType::rank() == 2)),
                           int> = 0>
typename FunctionViewType::value_type
interpolate_function(const PolynomialViewType &polynomial,
                     const FunctionViewType &function) {

  using ExecSpace = typename PolynomialViewType::execution_space;

  static_assert(std::is_same<typename PolynomialViewType::execution_space,
                             typename FunctionViewType::execution_space>::value,
                "Polynomial and function must have the same execution space");

  const int N = polynomial.extent(0);
  using T = typename FunctionViewType::value_type;

  T result(0.0);

  impl::InterpolateFunctor functor(polynomial, function);

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2> >({ 0, 0 }, { N, N }),
      functor, Kokkos::Sum<T>(result));

  return result;
}

template <typename PolynomialViewType, typename FunctionViewType,
          std::enable_if_t<((PolynomialViewType::rank() == 3) &&
                            (FunctionViewType::rank() == 3)),
                           int> = 0>
typename FunctionViewType::value_type
interpolate_function(const PolynomialViewType &polynomial,
                     const FunctionViewType &function) {

  using ExecSpace = typename PolynomialViewType::execution_space;

  static_assert(std::is_same<typename PolynomialViewType::execution_space,
                             typename FunctionViewType::execution_space>::value,
                "Polynomial and function must have the same execution space");

  const int N = polynomial.extent(0);
  using T = typename FunctionViewType::value_type;

  T result(0.0);
  impl::InterpolateFunctor functor(polynomial, function);

  Kokkos::parallel_reduce(Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3> >(
                              { 0, 0, 0 }, { N, N, N }),
                          functor, Kokkos::Sum<T>(result));

  return result;
}

// 2D version - 4D views
template <typename ChunkIndex, typename PolynomialViewType,
          typename FunctionViewType, typename ResultType,
          std::enable_if_t<PolynomialViewType::rank() == 4 &&
                               FunctionViewType::rank() == 4,
                           int> = 0>
KOKKOS_FUNCTION void interpolate_function(const ChunkIndex &chunk_index,
                                          const PolynomialViewType &polynomial,
                                          const FunctionViewType &function,
                                          ResultType &result) {

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

  const auto &team = chunk_index.get_policy_index();
  const int number_of_elements = result.extent(0);

  // // Initialize result
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, number_of_elements),
                       [&](const int &ielement) {
                         result(ielement, 0) = 0.0;
                         result(ielement, 1) = 0.0;
                       });

  team.team_barrier();

  const int ncomponents = function.extent(3);

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndex::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;

        for (int icomponent = 0; icomponent < ncomponents; ++icomponent) {
          type_real polynomial_value =
              polynomial(ielement, index.iz, index.ix, icomponent);
          type_real function_value =
              function(ielement, index.iz, index.ix, icomponent);
          Kokkos::atomic_add(&result(ielement, icomponent),
                             polynomial_value * function_value);
        }
      });

  return;
}

// 3D version - 5D views
template <typename ChunkIndex, typename PolynomialViewType,
          typename FunctionViewType, typename ResultType,
          std::enable_if_t<PolynomialViewType::rank() == 5 &&
                               FunctionViewType::rank() == 5,
                           int> = 0>
KOKKOS_FUNCTION void interpolate_function(const ChunkIndex &chunk_index,
                                          const PolynomialViewType &polynomial,
                                          const FunctionViewType &function,
                                          ResultType &result) {

  static_assert(ResultType::rank() == 2, "Result must be 2D views");

#ifndef NDEBUG

  if (polynomial.extent(0) != function.extent(0) ||
      polynomial.extent(1) != function.extent(1)) {
    Kokkos::abort("Polynomial and function must have the same size");
  }

  if (polynomial.extent(0) != result.extent(0)) {
    Kokkos::abort("Polynomial and result must have the same size");
  }

  if (function.extent(4) != result.extent(1)) {
    Kokkos::abort(
        "Function and result must have the same number of components");
  }
#endif

  const auto &team = chunk_index.get_policy_index();
  const int number_of_elements = result.extent(0);

  // // Initialize result
  const int ncomponents = function.extent(4);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, number_of_elements),
                       [&](const int &ielement) {
                         for (int icomp = 0; icomp < ncomponents; ++icomp) {
                           result(ielement, icomp) = 0.0;
                         }
                       });

  team.team_barrier();

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndex::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;

        for (int icomponent = 0; icomponent < ncomponents; ++icomponent) {
          type_real polynomial_value =
              polynomial(ielement, index.iz, index.iy, index.ix, icomponent);
          type_real function_value =
              function(ielement, index.iz, index.iy, index.ix, icomponent);
          Kokkos::atomic_add(&result(ielement, icomponent),
                             polynomial_value * function_value);
        }
      });

  return;
}

} // namespace algorithms
} // namespace specfem
