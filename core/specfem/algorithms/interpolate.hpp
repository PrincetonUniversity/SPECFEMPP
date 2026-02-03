#pragma once

#include "kokkos_abstractions.h"
#include "specfem/datatype.hpp"
#include "specfem/execution.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

/**
 * @file interpolate.hpp
 * @brief Algorithms for interpolating functions using polynomial basis
 * @ingroup AlgorithmsInterpolate
 */

namespace specfem {
namespace algorithms {

/// @brief Implementation details
namespace impl {
/**
 * @brief Functor for performing polynomial interpolation
 *
 * @tparam PolynomialViewType Type of the polynomial view
 * @tparam FunctionViewType Type of the function view
 */
template <typename PolynomialViewType, typename FunctionViewType>
struct InterpolateFunctor {
  const PolynomialViewType polynomial; ///< Polynomial basis functions
  const FunctionViewType function;     ///< Function values to interpolate

  constexpr static int rank =
      PolynomialViewType::rank(); ///< Rank of the polynomial view

  /**
   * @brief Constructor
   *
   * @param polynomial Polynomial basis functions
   * @param function Function values to interpolate
   */
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

/**
 * @brief Interpolate a 2D function using polynomial basis
 *
 * @ingroup AlgorithmsInterpolate
 * @tparam PolynomialViewType Type of the polynomial view (must be 2D)
 * @tparam FunctionViewType Type of the function view (must be 2D)
 * @param polynomial Polynomial basis functions
 * @param function Function values to interpolate
 * @return Interpolated function value
 */
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

/**
 * @brief Interpolate a 3D function using polynomial basis
 *
 * @ingroup AlgorithmsInterpolate
 * @tparam PolynomialViewType Type of the polynomial view (must be 3D)
 * @tparam FunctionViewType Type of the function view (must be 3D)
 * @param polynomial Polynomial basis functions
 * @param function Function values to interpolate
 * @return Interpolated function value
 */
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

/**
 * @brief Interpolate a 2D function using polynomial basis (Chunk version)
 *
 * @ingroup AlgorithmsInterpolate
 * @tparam ChunkIndex Chunk index type
 * @tparam PolynomialViewType Type of the polynomial view (rank 4)
 * @tparam FunctionViewType Type of the function view (rank 4)
 * @tparam ResultType Type of the result view (rank 2)
 * @param chunk_index Chunk index
 * @param polynomial Polynomial basis functions
 * @param function Function values to interpolate
 * @param result Result view
 */
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

/**
 * @brief Interpolate a 3D function using polynomial basis (Chunk version)
 *
 * @ingroup AlgorithmsInterpolate
 * @tparam ChunkIndex Chunk index type
 * @tparam PolynomialViewType Type of the polynomial view (rank 5)
 * @tparam FunctionViewType Type of the function view (rank 5)
 * @tparam ResultType Type of the result view (rank 2)
 * @param chunk_index Chunk index
 * @param polynomial Polynomial basis functions
 * @param function Function values to interpolate
 * @param result Result view
 */
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

/**
 * @defgroup AlgorithmsInterpolate
 *
 */
