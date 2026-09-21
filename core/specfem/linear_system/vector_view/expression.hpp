#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/tpetra_types.hpp"
#include <cstddef>
#include <type_traits>

namespace specfem {
namespace linear_system {

class VectorSpace;

/**
 * @brief One scaled vector of a sum: `alpha * vector`.
 *
 * Borrows its vector. `space` is the @ref specfem::linear_system::VectorSpace
 * the vector was drawn from, or null for a raw `vector_type` such as a lumped
 * mass vector; it is what lets an expression with no assignment target find
 * scratch.
 */
struct VectorTerm {
  specfem::linear_system::scalar_type alpha;         ///< Coefficient
  const specfem::linear_system::vector_type *vector; ///< Borrowed vector
  const specfem::linear_system::VectorSpace *space;  ///< Owning space, or null
};

/// An ordered sum of @ref VectorTerm
template <std::size_t N> struct Sum {
  VectorTerm terms[N == 0 ? 1 : N];      ///< Summands, in written order
  constexpr static std::size_t size = N; ///< Number of summands
};

/// `matrix * inner` -- one sparse matrix applied to a sum
template <typename Inner> struct MatrixProduct {
  const specfem::linear_system::crs_matrix_type *matrix; ///< Borrowed matrix
  Inner inner;                                           ///< Operand
};

/// `diag(vector) * inner` -- one diagonal matrix applied to a sum
template <typename Inner> struct DiagProduct {
  const specfem::linear_system::vector_type *diagonal; ///< Borrowed entries
  Inner inner;                                         ///< Operand
};

/**
 * @brief A sum plus one product, as `b - A * x` produces.
 *
 * The grammar carries at most one product; a second one is a compile error
 * naming the fix.
 *
 * @tparam N Summands in the plain part
 * @tparam Product @ref MatrixProduct or @ref DiagProduct
 */
template <std::size_t N, typename Product> struct Expression {
  Sum<N> sum;      ///< Plain part, in written order
  Product product; ///< Product part, applied after the plain part
};

} // namespace linear_system

namespace linear_system_impl {

/// Whether `T` is a @ref Sum
template <typename T> struct is_sum : std::false_type {};
template <std::size_t N>
struct is_sum<specfem::linear_system::Sum<N>> : std::true_type {};

/// Whether `T` is a @ref MatrixProduct or a @ref DiagProduct
template <typename T> struct is_product : std::false_type {};
template <typename Inner>
struct is_product<specfem::linear_system::MatrixProduct<Inner>>
    : std::true_type {};
template <typename Inner>
struct is_product<specfem::linear_system::DiagProduct<Inner>> : std::true_type {
};

/// Whether `T` is an @ref Expression
template <typename T> struct is_expression : std::false_type {};
template <std::size_t N, typename Product>
struct is_expression<specfem::linear_system::Expression<N, Product>>
    : std::true_type {};

/// Concatenate two sums, preserving order
template <std::size_t N, std::size_t M>
constexpr specfem::linear_system::Sum<N + M>
concat(const specfem::linear_system::Sum<N> &left,
       const specfem::linear_system::Sum<M> &right) {
  specfem::linear_system::Sum<N + M> result{};
  for (std::size_t i = 0; i < N; ++i) {
    result.terms[i] = left.terms[i];
  }
  for (std::size_t i = 0; i < M; ++i) {
    result.terms[N + i] = right.terms[i];
  }
  return result;
}

/// Scale every term of a sum
template <std::size_t N>
constexpr specfem::linear_system::Sum<N>
scale(const specfem::linear_system::Sum<N> &sum,
      const specfem::linear_system::scalar_type alpha) {
  specfem::linear_system::Sum<N> result = sum;
  for (std::size_t i = 0; i < N; ++i) {
    result.terms[i].alpha *= alpha;
  }
  return result;
}

/// Scale a product by scaling its operand
template <typename Inner>
constexpr specfem::linear_system::MatrixProduct<Inner>
scale(const specfem::linear_system::MatrixProduct<Inner> &product,
      const specfem::linear_system::scalar_type alpha) {
  return { product.matrix, scale(product.inner, alpha) };
}

/// Scale a product by scaling its operand
template <typename Inner>
constexpr specfem::linear_system::DiagProduct<Inner>
scale(const specfem::linear_system::DiagProduct<Inner> &product,
      const specfem::linear_system::scalar_type alpha) {
  return { product.diagonal, scale(product.inner, alpha) };
}

/// The space a sum draws scratch from, or null when every term is a raw vector
template <std::size_t N>
constexpr const specfem::linear_system::VectorSpace *
space_of(const specfem::linear_system::Sum<N> &sum) {
  for (std::size_t i = 0; i < N; ++i) {
    if (sum.terms[i].space != nullptr) {
      return sum.terms[i].space;
    }
  }
  return nullptr;
}

template <typename Inner>
constexpr const specfem::linear_system::VectorSpace *
space_of(const specfem::linear_system::MatrixProduct<Inner> &product) {
  return space_of(product.inner);
}

template <typename Inner>
constexpr const specfem::linear_system::VectorSpace *
space_of(const specfem::linear_system::DiagProduct<Inner> &product) {
  return space_of(product.inner);
}

template <std::size_t N, typename Product>
constexpr const specfem::linear_system::VectorSpace *
space_of(const specfem::linear_system::Expression<N, Product> &expression) {
  const auto *space = space_of(expression.sum);
  return space != nullptr ? space : space_of(expression.product);
}

/// Whether any term of an expression borrows `vector`
template <std::size_t N>
constexpr bool aliases(const specfem::linear_system::Sum<N> &sum,
                       const specfem::linear_system::vector_type *vector) {
  for (std::size_t i = 0; i < N; ++i) {
    if (sum.terms[i].vector == vector) {
      return true;
    }
  }
  return false;
}

template <typename Inner>
constexpr bool
aliases(const specfem::linear_system::MatrixProduct<Inner> &product,
        const specfem::linear_system::vector_type *vector) {
  return aliases(product.inner, vector);
}

template <typename Inner>
constexpr bool
aliases(const specfem::linear_system::DiagProduct<Inner> &product,
        const specfem::linear_system::vector_type *vector) {
  return aliases(product.inner, vector) || product.diagonal == vector;
}

template <std::size_t N, typename Product>
constexpr bool
aliases(const specfem::linear_system::Expression<N, Product> &expression,
        const specfem::linear_system::vector_type *vector) {
  return aliases(expression.sum, vector) || aliases(expression.product, vector);
}

} // namespace linear_system_impl
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
