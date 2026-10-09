#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/tpetra_types.hpp"
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace specfem {
namespace linear_system {

class VectorSpace;
class VectorView;

/**
 * @brief One scaled vector of an expression: `alpha * vector`.
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

/**
 * @brief An ordered sum of scaled vectors plus any number of operator
 * products.
 *
 * Every expression the grammar builds is one of these: `2 * u + 3 * v` carries
 * two terms and no products, `A * u` carries no terms and one product, and
 * `b - A * u + f` carries both. Products are applied after the plain terms, in
 * written order.
 *
 * @tparam N Number of scaled vectors
 * @tparam Products @ref MatrixProduct and @ref DiagProduct types, in order
 */
template <std::size_t N, typename... Products> struct Expression {
  std::array<VectorTerm, N> terms;  ///< Scaled vectors, in written order
  std::tuple<Products...> products; ///< Operator products, in written order

  constexpr static std::size_t size = N; ///< Number of scaled vectors
  constexpr static std::size_t num_products =
      sizeof...(Products); ///< Number of operator products
};

/// `matrix * inner` -- one sparse matrix applied to an expression
template <typename Inner> struct MatrixProduct {
  const specfem::linear_system::crs_matrix_type *matrix; ///< Borrowed matrix
  Inner inner;                                           ///< Operand
};

/// `diag(vector) * inner` -- one diagonal matrix applied to an expression
template <typename Inner> struct DiagProduct {
  const specfem::linear_system::vector_type *diagonal; ///< Borrowed entries
  Inner inner;                                         ///< Operand
};

} // namespace linear_system

namespace linear_system_impl {

/// Whether `T` is an @ref specfem::linear_system::Expression
template <typename T> struct is_expression : std::false_type {};
template <std::size_t N, typename... Products>
struct is_expression<specfem::linear_system::Expression<N, Products...>>
    : std::true_type {};

/// Whether `T` is a @ref specfem::linear_system::MatrixProduct or a
/// @ref specfem::linear_system::DiagProduct
template <typename T> struct is_product : std::false_type {};
template <typename Inner>
struct is_product<specfem::linear_system::MatrixProduct<Inner>>
    : std::true_type {};
template <typename Inner>
struct is_product<specfem::linear_system::DiagProduct<Inner>> : std::true_type {
};

/// Concatenate two expressions, preserving the order of both
template <std::size_t N, typename... Left, std::size_t M, typename... Right>
constexpr auto
join(const specfem::linear_system::Expression<N, Left...> &left,
     const specfem::linear_system::Expression<M, Right...> &right) {
  specfem::linear_system::Expression<N + M, Left..., Right...> result{};
  for (std::size_t i = 0; i < N; ++i) {
    result.terms[i] = left.terms[i];
  }
  for (std::size_t i = 0; i < M; ++i) {
    result.terms[N + i] = right.terms[i];
  }
  result.products = std::tuple_cat(left.products, right.products);
  return result;
}

template <typename Inner>
constexpr auto
scale_product(const specfem::linear_system::MatrixProduct<Inner> &product,
              specfem::linear_system::scalar_type alpha);
template <typename Inner>
constexpr auto
scale_product(const specfem::linear_system::DiagProduct<Inner> &product,
              specfem::linear_system::scalar_type alpha);

/// Scale every term and every product of an expression
template <std::size_t N, typename... Products>
constexpr auto
scale(const specfem::linear_system::Expression<N, Products...> &expression,
      const specfem::linear_system::scalar_type alpha) {
  specfem::linear_system::Expression<N, Products...> result = expression;
  for (std::size_t i = 0; i < N; ++i) {
    result.terms[i].alpha *= alpha;
  }
  result.products = std::apply(
      [alpha](const auto &...product) {
        return std::make_tuple(scale_product(product, alpha)...);
      },
      expression.products);
  return result;
}

/// A product is scaled by scaling its operand, so the coefficient reaches the
/// `alpha` the Tpetra call already takes
template <typename Inner>
constexpr auto
scale_product(const specfem::linear_system::MatrixProduct<Inner> &product,
              const specfem::linear_system::scalar_type alpha) {
  return specfem::linear_system::MatrixProduct<Inner>{
    product.matrix, scale(product.inner, alpha)
  };
}

template <typename Inner>
constexpr auto
scale_product(const specfem::linear_system::DiagProduct<Inner> &product,
              const specfem::linear_system::scalar_type alpha) {
  return specfem::linear_system::DiagProduct<Inner>{
    product.diagonal, scale(product.inner, alpha)
  };
}

template <typename Inner>
constexpr const specfem::linear_system::VectorSpace *
space_of_product(const specfem::linear_system::MatrixProduct<Inner> &product);
template <typename Inner>
constexpr const specfem::linear_system::VectorSpace *
space_of_product(const specfem::linear_system::DiagProduct<Inner> &product);

/// The space an expression draws scratch from, or null when every vector in it
/// is a raw one
template <std::size_t N, typename... Products>
constexpr const specfem::linear_system::VectorSpace *
space_of(const specfem::linear_system::Expression<N, Products...> &expression) {
  for (std::size_t i = 0; i < N; ++i) {
    if (expression.terms[i].space != nullptr) {
      return expression.terms[i].space;
    }
  }
  const specfem::linear_system::VectorSpace *found = nullptr;
  std::apply(
      [&found](const auto &...product) {
        ((found = found != nullptr ? found : space_of_product(product)), ...);
      },
      expression.products);
  return found;
}

template <typename Inner>
constexpr const specfem::linear_system::VectorSpace *
space_of_product(const specfem::linear_system::MatrixProduct<Inner> &product) {
  return space_of(product.inner);
}

template <typename Inner>
constexpr const specfem::linear_system::VectorSpace *
space_of_product(const specfem::linear_system::DiagProduct<Inner> &product) {
  return space_of(product.inner);
}

template <typename Inner>
constexpr bool
aliases_product(const specfem::linear_system::MatrixProduct<Inner> &product,
                const specfem::linear_system::vector_type *vector);
template <typename Inner>
constexpr bool
aliases_product(const specfem::linear_system::DiagProduct<Inner> &product,
                const specfem::linear_system::vector_type *vector);

/// Whether any vector an expression reads is `vector`
template <std::size_t N, typename... Products>
constexpr bool
aliases(const specfem::linear_system::Expression<N, Products...> &expression,
        const specfem::linear_system::vector_type *vector) {
  for (std::size_t i = 0; i < N; ++i) {
    if (expression.terms[i].vector == vector) {
      return true;
    }
  }
  bool found = false;
  std::apply(
      [&found, vector](const auto &...product) {
        ((found = found || aliases_product(product, vector)), ...);
      },
      expression.products);
  return found;
}

template <typename Inner>
constexpr bool
aliases_product(const specfem::linear_system::MatrixProduct<Inner> &product,
                const specfem::linear_system::vector_type *vector) {
  return aliases(product.inner, vector);
}

template <typename Inner>
constexpr bool
aliases_product(const specfem::linear_system::DiagProduct<Inner> &product,
                const specfem::linear_system::vector_type *vector) {
  return aliases(product.inner, vector) || product.diagonal == vector;
}

} // namespace linear_system_impl

namespace linear_system {

/// An expression the vector grammar can evaluate
template <typename T>
concept VectorExpression =
    specfem::linear_system_impl::is_expression<std::remove_cvref_t<T>>::value;

/**
 * @brief Anything the grammar's operators accept.
 *
 * An expression, an operator product, a @ref VectorView, or a raw
 * `vector_type`. Everything else -- a matrix on its own, a bare scalar, a
 * matrix times a matrix -- simply fails to match, which is what keeps
 * non-algebraic spellings out without a single rejection overload.
 */
template <typename T>
concept VectorOperand =
    VectorExpression<T> ||
    specfem::linear_system_impl::is_product<std::remove_cvref_t<T>>::value ||
    std::is_same_v<std::remove_cvref_t<T>, VectorView> ||
    std::is_same_v<std::remove_cvref_t<T>, vector_type>;

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
