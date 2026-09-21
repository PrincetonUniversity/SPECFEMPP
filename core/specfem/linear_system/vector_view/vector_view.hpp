#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/operands.hpp"
#include "specfem/linear_system/tpetra_types.hpp"
#include "specfem/linear_system/vector_view/expression.hpp"
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCP.hpp>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace specfem {
namespace linear_system {

class VectorView;

/**
 * @brief The dof map a set of solver vectors share, and the scratch they
 * evaluate expressions in.
 *
 * Vectors are created from a space rather than from a map so that an
 * expression can borrow a temporary without allocating one per statement:
 *
 * @code
 * specfem::linear_system::VectorSpace space(fe.owned_map());
 * auto u = space.vector();
 * auto v = space.vector();
 * u = 2 * u + v;
 * @endcode
 *
 * The pool grows to the deepest expression ever evaluated -- two vectors for
 * the implicit Newmark update -- and is reused from then on. A space must
 * outlive every vector drawn from it.
 */
class VectorSpace {
public:
  /**
   * @brief Bind a space to a dof map.
   *
   * @param map Row map of every vector drawn from this space
   */
  explicit VectorSpace(Teuchos::RCP<const map_type> map)
      : map_(std::move(map)) {}

  VectorSpace(const VectorSpace &) = delete;
  VectorSpace &operator=(const VectorSpace &) = delete;

  /// A new zero-initialised vector bound to this space
  VectorView vector() const;

  /// Row map every vector of this space is built on
  Teuchos::RCP<const map_type> map() const { return map_; }

  /// Vectors currently held by the scratch pool
  std::size_t scratch_size() const { return scratch_.size(); }

  /**
   * @brief Borrowed scratch vector, returned to the pool at scope exit.
   *
   * Not constructed directly; see @ref VectorSpace::borrow.
   */
  class Scratch {
  public:
    Scratch(const VectorSpace &space, const std::size_t slot)
        : space_(space), slot_(slot) {}

    ~Scratch() { space_.release(); }

    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;

    /// The borrowed vector
    vector_type &vector() const { return *space_.scratch_[slot_]; }

  private:
    const VectorSpace &space_; ///< Pool the slot belongs to
    const std::size_t slot_;   ///< Index into the pool
  };

  /**
   * @brief Borrow a scratch vector for the duration of one expression.
   *
   * Borrows nest but never interleave -- every @ref Scratch is a scoped local
   * and the type is neither copyable nor movable -- so the pool is a stack and
   * only grows when an expression nests deeper than any before it.
   */
  Scratch borrow() const {
    if (depth_ == scratch_.size()) {
      scratch_.push_back(Teuchos::rcp(new vector_type(map_)));
    }
    return Scratch(*this, depth_++);
  }

private:
  /// Return the innermost borrowed slot to the pool
  void release() const { --depth_; }

  Teuchos::RCP<const map_type> map_; ///< Row map of the space

  // Mutable so that an expression over const vectors can still borrow.
  mutable std::vector<Teuchos::RCP<vector_type>> scratch_; ///< Pool
  mutable std::size_t depth_ = 0; ///< Slots currently borrowed
};

/**
 * @brief An owning solver vector that reads as a MATLAB variable.
 *
 * Assignment and accumulation take an expression over other vectors, and the
 * whole expression is flattened into `Tpetra` calls at the assignment -- no
 * temporary vector per operator:
 *
 * @code
 * b  = f;
 * b += diag(mass) * (c0 * u + c1 * v + c2 * a);
 * b += damping * (c3 * u + c4 * v + c5 * a);
 * a_new = c0 * (u_new - u - dt * v) - c2 * a;
 * const type_real residual = norm2(b - system * u_new);
 * @endcode
 *
 * The supported grammar is a scaled sum of any length, optionally plus one
 * matrix or diagonal product. Anything outside it -- a scaled operator, a
 * nested product, two products in one expression -- is a compile error naming
 * the fix rather than a silent allocation.
 *
 * A view borrows its @ref VectorSpace and must not outlive it.
 */
class VectorView {
public:
  /**
   * @brief Bind a view to an existing vector.
   *
   * Prefer @ref VectorSpace::vector, which allocates the vector for you.
   *
   * @param space Space the vector belongs to; must outlive the view
   * @param vector Vector to own
   */
  VectorView(const VectorSpace &space, Teuchos::RCP<vector_type> vector)
      : space_(&space), vector_(std::move(vector)) {}

  // Assignment copies values, so a copy constructor that shared storage
  // instead would give `VectorView b = u;` and `b = u;` opposite meanings.
  VectorView(const VectorView &) = delete;
  VectorView(VectorView &&) = default;
  VectorView &operator=(VectorView &&) = default;

  /// The underlying Tpetra vector
  vector_type &vector() const { return *vector_; }

  /// The underlying Tpetra vector, for Belos and Ifpack2
  const Teuchos::RCP<vector_type> &rcp() const { return vector_; }

  /// Space this vector was drawn from
  const VectorSpace &space() const { return *space_; }

  /// Overwrite every entry with one value
  VectorView &operator=(const scalar_type value) {
    vector_->putScalar(value);
    return *this;
  }

  /// Copy another vector's values (not its identity)
  VectorView &operator=(const VectorView &other);

  /// Assign an expression
  template <VectorOperand Operand>
  VectorView &operator=(const Operand &operand);

  /// Accumulate an expression
  template <VectorOperand Operand>
  VectorView &operator+=(const Operand &operand);

  /// Accumulate another vector
  VectorView &operator+=(const VectorView &other);

private:
  const VectorSpace *space_;         ///< Borrowed space
  Teuchos::RCP<vector_type> vector_; ///< Owned storage

  friend void swap(VectorView &left, VectorView &right) {
    std::swap(left.space_, right.space_);
    std::swap(left.vector_, right.vector_);
  }
};

inline VectorView VectorSpace::vector() const {
  return VectorView(*this, Teuchos::rcp(new vector_type(map_)));
}

} // namespace linear_system

namespace linear_system_impl {

/// A view as a one-term expression
inline specfem::linear_system::Expression<1>
as_expression(const specfem::linear_system::VectorView &vector) {
  return { { specfem::linear_system::VectorTerm{
               static_cast<specfem::linear_system::scalar_type>(1),
               &vector.vector(), &vector.space() } },
           {} };
}

/// A raw Tpetra vector as a one-term expression; carries no space
inline specfem::linear_system::Expression<1>
as_expression(const specfem::linear_system::vector_type &vector) {
  return { { specfem::linear_system::VectorTerm{
               static_cast<specfem::linear_system::scalar_type>(1), &vector,
               nullptr } },
           {} };
}

/// A product as an expression of no plain terms
template <typename Inner>
constexpr auto
as_expression(const specfem::linear_system::MatrixProduct<Inner> &product) {
  return specfem::linear_system::Expression<
      0, specfem::linear_system::MatrixProduct<Inner>>{ {}, { product } };
}

template <typename Inner>
constexpr auto
as_expression(const specfem::linear_system::DiagProduct<Inner> &product) {
  return specfem::linear_system::Expression<
      0, specfem::linear_system::DiagProduct<Inner>>{ {}, { product } };
}

/// An expression is already an expression
template <std::size_t N, typename... Products>
constexpr auto as_expression(
    const specfem::linear_system::Expression<N, Products...> &expression) {
  return expression;
}

/// The expression type `as_expression` produces for `T`
template <typename T>
using expression_type_t = decltype(as_expression(std::declval<const T &>()));

template <std::size_t N, typename... Products>
void emit(specfem::linear_system::vector_type &target,
          const specfem::linear_system::Expression<N, Products...> &expression,
          bool &overwrite, const specfem::linear_system::VectorSpace &space);

/**
 * @brief Apply one matrix product into `target`.
 *
 * `Tpetra::CrsMatrix::apply` computes `Y = beta * Y + alpha * A * X`, so both
 * the product's own coefficient and the accumulation ride the call. A single
 * unscaled-or-scaled plain operand is applied straight from its vector;
 * anything longer is materialised into scratch first.
 */
template <typename Inner>
void emit_product(specfem::linear_system::vector_type &target,
                  const specfem::linear_system::MatrixProduct<Inner> &product,
                  bool &overwrite,
                  const specfem::linear_system::VectorSpace &space) {
  using scalar_type = specfem::linear_system::scalar_type;

  const scalar_type beta =
      overwrite ? static_cast<scalar_type>(0) : static_cast<scalar_type>(1);

  if constexpr (Inner::size == 1 && Inner::num_products == 0) {
    product.matrix->apply(*product.inner.terms[0].vector, target,
                          Teuchos::NO_TRANS, product.inner.terms[0].alpha,
                          beta);
  } else {
    const auto scratch = space.borrow();
    bool scratch_overwrite = true;
    emit(scratch.vector(), product.inner, scratch_overwrite, space);
    product.matrix->apply(scratch.vector(), target, Teuchos::NO_TRANS,
                          static_cast<scalar_type>(1), beta);
  }
  overwrite = false;
}

/**
 * @brief Apply one diagonal product into `target`.
 *
 * `elementWiseMultiply` computes `this = gamma * this + alpha * (d .* x)`, so
 * it carries the coefficient and the accumulation the same way.
 */
template <typename Inner>
void emit_product(specfem::linear_system::vector_type &target,
                  const specfem::linear_system::DiagProduct<Inner> &product,
                  bool &overwrite,
                  const specfem::linear_system::VectorSpace &space) {
  using scalar_type = specfem::linear_system::scalar_type;

  const scalar_type gamma =
      overwrite ? static_cast<scalar_type>(0) : static_cast<scalar_type>(1);

  if constexpr (Inner::size == 1 && Inner::num_products == 0) {
    target.elementWiseMultiply(product.inner.terms[0].alpha, *product.diagonal,
                               *product.inner.terms[0].vector, gamma);
  } else {
    const auto scratch = space.borrow();
    bool scratch_overwrite = true;
    emit(scratch.vector(), product.inner, scratch_overwrite, space);
    target.elementWiseMultiply(static_cast<scalar_type>(1), *product.diagonal,
                               scratch.vector(), gamma);
  }
  overwrite = false;
}

/**
 * @brief Write an expression into `target`.
 *
 * Plain terms go two at a time, which is what `Tpetra`'s three- and
 * five-argument `update` take, then each product accumulates in written order.
 * `overwrite` is true while `target`'s current contents are still to be
 * discarded, and is cleared by the first write.
 */
template <std::size_t N, typename... Products>
void emit(specfem::linear_system::vector_type &target,
          const specfem::linear_system::Expression<N, Products...> &expression,
          bool &overwrite, const specfem::linear_system::VectorSpace &space) {
  using scalar_type = specfem::linear_system::scalar_type;

  std::size_t i = 0;
  while (i + 1 < N) {
    target.update(
        expression.terms[i].alpha, *expression.terms[i].vector,
        expression.terms[i + 1].alpha, *expression.terms[i + 1].vector,
        overwrite ? static_cast<scalar_type>(0) : static_cast<scalar_type>(1));
    overwrite = false;
    i += 2;
  }
  if (i < N) {
    target.update(expression.terms[i].alpha, *expression.terms[i].vector,
                  overwrite ? static_cast<scalar_type>(0)
                            : static_cast<scalar_type>(1));
    overwrite = false;
  }

  std::apply(
      [&](const auto &...product) {
        (emit_product(target, product, overwrite, space), ...);
      },
      expression.products);

  // An expression with no terms and no products -- which the grammar cannot
  // build, but the type admits -- still has to leave an assignment target
  // defined.
  if (overwrite) {
    target.putScalar(static_cast<scalar_type>(0));
    overwrite = false;
  }
}

/**
 * @brief Evaluate an expression into `target`.
 *
 * An expression that reads `target` is evaluated in scratch and copied back,
 * so `x = 2 * x + y` is correct rather than order-dependent, and so a product
 * never reaches `Tpetra` with its input aliasing its output.
 *
 * @param target Vector written
 * @param expression Expression to evaluate
 * @param overwrite Whether `target`'s current contents are discarded
 * @param space Space scratch is borrowed from
 */
template <std::size_t N, typename... Products>
void evaluate(
    specfem::linear_system::vector_type &target,
    const specfem::linear_system::Expression<N, Products...> &expression,
    const bool overwrite, const specfem::linear_system::VectorSpace &space) {
  using scalar_type = specfem::linear_system::scalar_type;

  if (aliases(expression, &target)) {
    const auto scratch = space.borrow();
    bool scratch_overwrite = true;
    emit(scratch.vector(), expression, scratch_overwrite, space);
    target.update(static_cast<scalar_type>(1), scratch.vector(),
                  overwrite ? static_cast<scalar_type>(0)
                            : static_cast<scalar_type>(1));
    return;
  }

  bool pending = overwrite;
  emit(target, expression, pending, space);
}

} // namespace linear_system_impl

namespace linear_system {

// ── Building expressions ───────────────────────────────────────────────────

/**
 * @brief Add any two operands: `x + y`.
 *
 * The whole additive grammar. Both sides are normalised to expressions and
 * concatenated, so a vector, a sum, a product and a sum-plus-products all
 * compose with each other and with themselves.
 */
template <VectorOperand Left, VectorOperand Right>
constexpr auto operator+(const Left &left, const Right &right) {
  return specfem::linear_system_impl::join(
      specfem::linear_system_impl::as_expression(left),
      specfem::linear_system_impl::as_expression(right));
}

/// Scale any operand: `alpha * x`
template <VectorOperand Operand>
constexpr auto operator*(const scalar_type alpha, const Operand &operand) {
  return specfem::linear_system_impl::scale(
      specfem::linear_system_impl::as_expression(operand), alpha);
}

/// Scale any operand: `x * alpha`
template <VectorOperand Operand>
constexpr auto operator*(const Operand &operand, const scalar_type alpha) {
  return alpha * operand;
}

/// Negate any operand: `-x`
template <VectorOperand Operand>
constexpr auto operator-(const Operand &operand) {
  return static_cast<scalar_type>(-1) * operand;
}

/// Subtract any two operands: `x - y`
template <VectorOperand Left, VectorOperand Right>
constexpr auto operator-(const Left &left, const Right &right) {
  return left + (-right);
}

// ── Products ───────────────────────────────────────────────────────────────

/// Apply a matrix: `A * x`
template <VectorOperand Operand>
constexpr auto operator*(const crs_matrix_type &matrix,
                         const Operand &operand) {
  return MatrixProduct<specfem::linear_system_impl::expression_type_t<Operand>>{
    &matrix, specfem::linear_system_impl::as_expression(operand)
  };
}

/// Apply a diagonal matrix: `diag(m) * x`
template <VectorOperand Operand>
constexpr auto operator*(const Diagonal diagonal, const Operand &operand) {
  return DiagProduct<specfem::linear_system_impl::expression_type_t<Operand>>{
    &diagonal.vector, specfem::linear_system_impl::as_expression(operand)
  };
}

/**
 * @brief Apply a scaled matrix: `alpha * A * x`.
 *
 * The coefficient folds into the operand, where it reaches the `alpha` that
 * `apply` already takes -- so this costs nothing over the unscaled spelling.
 */
template <VectorOperand Operand>
constexpr auto operator*(const ScaledMatrix scaled, const Operand &operand) {
  return scaled.matrix * (scaled.alpha * operand);
}

/// Apply a scaled diagonal matrix: `alpha * diag(m) * x`
template <VectorOperand Operand>
constexpr auto operator*(const ScaledDiagonal scaled, const Operand &operand) {
  return Diagonal{ scaled.vector } * (scaled.alpha * operand);
}

// ── Assignment ─────────────────────────────────────────────────────────────

inline VectorView &VectorView::operator=(const VectorView &other) {
  return *this = specfem::linear_system_impl::as_expression(other);
}

template <VectorOperand Operand>
VectorView &VectorView::operator=(const Operand &operand) {
  specfem::linear_system_impl::evaluate(
      *vector_, specfem::linear_system_impl::as_expression(operand), true,
      *space_);
  return *this;
}

template <VectorOperand Operand>
VectorView &VectorView::operator+=(const Operand &operand) {
  specfem::linear_system_impl::evaluate(
      *vector_, specfem::linear_system_impl::as_expression(operand), false,
      *space_);
  return *this;
}

inline VectorView &VectorView::operator+=(const VectorView &other) {
  return *this += specfem::linear_system_impl::as_expression(other);
}

/**
 * @brief View a vector as the diagonal of a matrix.
 *
 * The @ref VectorView overload of specfem::linear_system::diag, so that
 * `diag(mass)` resolves without qualification when `mass` is a view.
 *
 * @param vector Diagonal entries; must outlive the expression
 * @return Wrapper accepted by the product operators
 */
inline Diagonal diag(const VectorView &vector) {
  return Diagonal{ vector.vector() };
}

// ── Norms ──────────────────────────────────────────────────────────────────

/// Euclidean norm of a vector
inline type_real norm2(const VectorView &vector) {
  return static_cast<type_real>(vector.vector().norm2());
}

/**
 * @brief Euclidean norm of an expression.
 *
 * Evaluates into scratch borrowed from the expression's own space, so a
 * residual reads as one statement: `norm2(b - A * x)`.
 *
 * @param expression Expression to evaluate
 * @return \f$ \| \mathrm{expression} \|_2 \f$
 */
template <VectorOperand Operand>
  requires(!std::is_same_v<std::remove_cvref_t<Operand>, VectorView>)
type_real norm2(const Operand &operand) {
  const auto expression = specfem::linear_system_impl::as_expression(operand);
  const auto *space = specfem::linear_system_impl::space_of(expression);
  if (space == nullptr) {
    throw std::runtime_error(
        "specfem::linear_system::norm2: the expression holds no vector drawn "
        "from a VectorSpace, so there is nowhere to borrow scratch from.");
  }

  const auto scratch = space->borrow();
  specfem::linear_system_impl::evaluate(scratch.vector(), expression, true,
                                        *space);
  return static_cast<type_real>(scratch.vector().norm2());
}

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
