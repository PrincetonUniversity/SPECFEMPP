#pragma once

#include "Kokkos_Array.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::quadrature::compiletime::impl {
/**
 * @brief Compute polynomial coefficients for one Lagrange basis function.
 *
 * Returns the coefficients @f$ c_k @f$ such that
 * @f[
 *   L_{\text{basis\_idx}}(x) = \sum_{k=0}^{N-1} c_k \, x^k.
 * @f]
 *
 * @tparam N         Number of nodes.
 * @param  nodes     GLL node array of length @p N.
 * @param  basis_idx Index of the desired basis polynomial.
 * @return Array of @p N polynomial coefficients, lowest-degree first.
 */
template <typename Number, std::size_t N>
consteval Kokkos::Array<Number, N>
make_lagrange_coeffs_for_basis(const Kokkos::Array<Number, N> &nodes,
                               std::size_t basis_idx) {
  Kokkos::Array<Number, N> coeff{};
  coeff[0] = 1.f;
  std::size_t degree = 0;
  Number denom = 1.f;

  for (std::size_t j = 0; j < N; ++j) {
    if (j == basis_idx)
      continue;
    for (std::size_t k = degree + 1; k > 0; --k) {
      coeff[k] = coeff[k - 1] - nodes[j] * coeff[k];
    }
    coeff[0] = -nodes[j] * coeff[0];
    denom *= (nodes[basis_idx] - nodes[j]);
    ++degree;
  }

  for (std::size_t k = 0; k < N; ++k)
    coeff[k] /= denom;
  return coeff;
}

/**
 * @brief Build the full coefficient table for all Lagrange basis functions.
 *
 * @tparam N     Number of nodes.
 * @param  nodes GLL node array of length @p N.
 * @return NxN array where row @p i holds the coefficients of @f$ L_i @f$.
 */
template <typename Number, std::size_t N>
consteval Kokkos::Array<Kokkos::Array<Number, N>, N>
make_lagrange_coeff_table(const Kokkos::Array<Number, N> &nodes) {
  Kokkos::Array<Kokkos::Array<Number, N>, N> table{};
  for (std::size_t i = 0; i < N; ++i)
    table[i] = make_lagrange_coeffs_for_basis(nodes, i);
  return table;
}

/**
 * @brief Evaluate all five basis polynomials at @p x using Horner's method.
 * @param x  Evaluation point in @f$ [-1,1] @f$.
 * @param L  Output array; @c L[i] receives @f$ L_i(x) @f$.
 */
template <std::size_t N, typename TableNumberType, typename SampleNumberType>
KOKKOS_INLINE_FUNCTION constexpr void lagrange_eval_all(
    const Kokkos::Array<Kokkos::Array<TableNumberType, N>, N> &coeff_table,
    SampleNumberType x, Kokkos::Array<SampleNumberType, N> &L) {
  for (int i = 0; i < N; ++i) {
    float v = coeff_table[i][N - 1];
    for (int k = N - 2; k >= 0; --k)
      v = v * x + coeff_table[i][k];
    L[i] = v;
  }
}

template <typename Number, std::size_t N>
consteval Number lagrange_polynomial_orthogonality_error(
    const Kokkos::Array<Number, N> &nodes,
    const Kokkos::Array<Kokkos::Array<Number, N>, N> &coeff_table) {
  Number max_error = 0;
  Kokkos::Array<Number, N> eval_stores;

  for (std::size_t inode = 0; inode < N; ++inode) {

    // for each node, evaluate all L[i] and compare to Kronecker delta.

    Number x = nodes[inode];
    lagrange_eval_all(coeff_table, x, eval_stores);

    for (std::size_t ifunc = 0; ifunc < N; ++ifunc) {
      Number error = eval_stores[ifunc] - ((ifunc == inode) ? 1 : 0);

      // max_error = max(abs(error), max_error)
      if (error > max_error) {
        max_error = error;
      } else if (error < -max_error) {
        max_error = -error;
      }
    }
  }
  return max_error;
}

} // namespace specfem::quadrature::compiletime::impl
