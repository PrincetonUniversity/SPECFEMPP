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
  template <VectorExpression Expr> VectorView &operator=(const Expr &expr);

  /// Accumulate an expression
  template <VectorExpression Expr> VectorView &operator+=(const Expr &expr);

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

/// A single vector as a one-term sum
inline specfem::linear_system::Sum<1>
to_sum(const specfem::linear_system::VectorView &vector) {
  return specfem::linear_system::Sum<1>{
    { { static_cast<specfem::linear_system::scalar_type>(1), &vector.vector(),
        &vector.space() } }
  };
}

/// A raw Tpetra vector as a one-term sum; carries no space
inline specfem::linear_system::Sum<1>
to_sum(const specfem::linear_system::vector_type &vector) {
  return specfem::linear_system::Sum<1>{
    { { static_cast<specfem::linear_system::scalar_type>(1), &vector,
        nullptr } }
  };
}

/// A sum is already a sum
template <std::size_t N>
constexpr specfem::linear_system::Sum<N>
to_sum(const specfem::linear_system::Sum<N> &sum) {
  return sum;
}

/// The operand a scaling or negation applies to: a sum stays a sum, a product
/// stays a product, a vector becomes a one-term sum
template <typename T> constexpr auto as_operand(const T &operand) {
  if constexpr (is_product<T>::value) {
    return operand;
  } else {
    return to_sum(operand);
  }
}

/// The sum type `to_sum` produces for `T`
template <typename T>
using sum_type_t = decltype(to_sum(std::declval<const T &>()));

/**
 * @brief Write a sum into `target`.
 *
 * Emits terms two at a time, which is what `Tpetra`'s three- and five-argument
 * `update` take. `overwrite` is true while `target`'s current contents are
 * still to be discarded, and is cleared by the first emission.
 */
template <std::size_t N>
void emit(specfem::linear_system::vector_type &target,
          const specfem::linear_system::Sum<N> &sum, bool &overwrite,
          const specfem::linear_system::VectorSpace & /* space */) {
  using scalar_type = specfem::linear_system::scalar_type;

  std::size_t i = 0;
  while (i + 1 < N) {
    target.update(sum.terms[i].alpha, *sum.terms[i].vector,
                  sum.terms[i + 1].alpha, *sum.terms[i + 1].vector,
                  overwrite ? static_cast<scalar_type>(0)
                            : static_cast<scalar_type>(1));
    overwrite = false;
    i += 2;
  }
  if (i < N) {
    target.update(sum.terms[i].alpha, *sum.terms[i].vector,
                  overwrite ? static_cast<scalar_type>(0)
                            : static_cast<scalar_type>(1));
    overwrite = false;
  }
}

/**
 * @brief Write a matrix product into `target`.
 *
 * `Tpetra::CrsMatrix::apply` computes `Y = beta * Y + alpha * A * X`, so the
 * product accumulates without a second scratch vector. A single unscaled
 * operand is applied directly; anything longer is materialised first.
 */
template <typename Inner>
void emit(specfem::linear_system::vector_type &target,
          const specfem::linear_system::MatrixProduct<Inner> &product,
          bool &overwrite, const specfem::linear_system::VectorSpace &space) {
  using scalar_type = specfem::linear_system::scalar_type;

  const scalar_type beta =
      overwrite ? static_cast<scalar_type>(0) : static_cast<scalar_type>(1);

  if constexpr (Inner::size == 1) {
    product.matrix->apply(*product.inner.terms[0].vector, target,
                          Teuchos::NO_TRANS, product.inner.terms[0].alpha,
                          beta);
    overwrite = false;
    return;
  }

  const auto scratch = space.borrow();
  bool scratch_overwrite = true;
  emit(scratch.vector(), product.inner, scratch_overwrite, space);
  product.matrix->apply(scratch.vector(), target, Teuchos::NO_TRANS,
                        static_cast<scalar_type>(1), beta);
  overwrite = false;
}

/**
 * @brief Write a diagonal product into `target`.
 *
 * `elementWiseMultiply` computes `this = gamma * this + alpha * (d .* x)`, so
 * the product accumulates in place. The operand is always materialised: it
 * may not alias the target, and a scratch vector is the cheapest guarantee.
 */
template <typename Inner>
void emit(specfem::linear_system::vector_type &target,
          const specfem::linear_system::DiagProduct<Inner> &product,
          bool &overwrite, const specfem::linear_system::VectorSpace &space) {
  using scalar_type = specfem::linear_system::scalar_type;

  const scalar_type gamma =
      overwrite ? static_cast<scalar_type>(0) : static_cast<scalar_type>(1);

  if constexpr (Inner::size == 1) {
    target.elementWiseMultiply(product.inner.terms[0].alpha, *product.diagonal,
                               *product.inner.terms[0].vector, gamma);
    overwrite = false;
    return;
  }

  const auto scratch = space.borrow();
  bool scratch_overwrite = true;
  emit(scratch.vector(), product.inner, scratch_overwrite, space);
  target.elementWiseMultiply(static_cast<scalar_type>(1), *product.diagonal,
                             scratch.vector(), gamma);
  overwrite = false;
}

/// Write a sum-plus-product into `target`, plain part first
template <std::size_t N, typename Product>
void emit(specfem::linear_system::vector_type &target,
          const specfem::linear_system::Expression<N, Product> &expression,
          bool &overwrite, const specfem::linear_system::VectorSpace &space) {
  emit(target, expression.sum, overwrite, space);
  emit(target, expression.product, overwrite, space);
}

/**
 * @brief Evaluate any expression into `target`.
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
template <typename Expr>
void evaluate(specfem::linear_system::vector_type &target,
              const Expr &expression, const bool overwrite,
              const specfem::linear_system::VectorSpace &space) {
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

// ── Scaling and negation ───────────────────────────────────────────────────

/// Scale a vector, sum or product: `alpha * x`
template <VectorOperand Operand>
constexpr auto operator*(const scalar_type alpha, const Operand &operand) {
  return specfem::linear_system_impl::scale(
      specfem::linear_system_impl::as_operand(operand), alpha);
}

/// Scale a vector, sum or product: `x * alpha`
template <VectorOperand Operand>
constexpr auto operator*(const Operand &operand, const scalar_type alpha) {
  return alpha * operand;
}

/// Negate a vector, sum or product: `-x`
template <VectorOperand Operand>
constexpr auto operator-(const Operand &operand) {
  return static_cast<scalar_type>(-1) * operand;
}

// ── Sums ───────────────────────────────────────────────────────────────────

/// Add two vectors or sums
template <SumLike Left, SumLike Right>
constexpr auto operator+(const Left &left, const Right &right) {
  return specfem::linear_system_impl::concat(
      specfem::linear_system_impl::to_sum(left),
      specfem::linear_system_impl::to_sum(right));
}

// ── Products ───────────────────────────────────────────────────────────────

/// Apply a matrix: `A * x`
template <SumLike Operand>
constexpr auto operator*(const crs_matrix_type &matrix,
                         const Operand &operand) {
  return specfem::linear_system::MatrixProduct<
      specfem::linear_system_impl::sum_type_t<Operand>>{
    &matrix, specfem::linear_system_impl::to_sum(operand)
  };
}

/// Apply a diagonal matrix: `diag(m) * x`
template <SumLike Operand>
constexpr auto operator*(const Diagonal diagonal, const Operand &operand) {
  return specfem::linear_system::DiagProduct<
      specfem::linear_system_impl::sum_type_t<Operand>>{
    &diagonal.vector, specfem::linear_system_impl::to_sum(operand)
  };
}

// ── Sum plus product ───────────────────────────────────────────────────────

/// `x + A * y`
template <SumLike Left, VectorProduct Product>
constexpr auto operator+(const Left &left, const Product &product) {
  return specfem::linear_system::Expression<
      specfem::linear_system_impl::sum_type_t<Left>::size, Product>{
    specfem::linear_system_impl::to_sum(left), product
  };
}

/// `A * y + x`
template <VectorProduct Product, SumLike Right>
constexpr auto operator+(const Product &product, const Right &right) {
  return right + product;
}

// ── Subtraction ────────────────────────────────────────────────────────────

/**
 * @brief Subtract any two operands: `x - y`.
 *
 * Every difference in the grammar is `left + (-right)`, and negation is
 * defined for both operand kinds, so one definition covers sum minus sum, sum
 * minus product, and product minus sum.
 */
template <VectorOperand Left, VectorOperand Right>
constexpr auto operator-(const Left &left, const Right &right) {
  return left + (-right);
}

// ── Rejected forms ─────────────────────────────────────────────────────────

/**
 * @brief Rejects `alpha * A * x`; scale the operand instead.
 *
 * Returns a grammar type rather than `void` so the `static_assert` is the only
 * diagnostic the caller sees.
 */
template <VectorOperand Operand>
specfem::linear_system::Sum<1> operator*(const ScaledMatrix, const Operand &) {
  static_assert(
      specfem::linear_system_impl::always_false_v<Operand>,
      "specfem::linear_system: a scaled matrix cannot be applied to a vector "
      "expression. Scale the operand instead -- A * (alpha * x) -- which is "
      "the same arithmetic in one fewer pass.");
  return {};
}

/// Rejects `alpha * diag(m) * x`; scale the operand instead
template <VectorOperand Operand>
specfem::linear_system::Sum<1> operator*(const ScaledDiagonal,
                                         const Operand &) {
  static_assert(
      specfem::linear_system_impl::always_false_v<Operand>,
      "specfem::linear_system: a scaled diagonal cannot be applied to a "
      "vector expression. Scale the operand instead: diag(m) * (alpha * x).");
  return {};
}

/// Rejects a nested product such as `A * (B * x)`
template <typename Outer, VectorProduct Inner>
  requires(VectorOperand<Outer> || std::is_same_v<Outer, crs_matrix_type> ||
           std::is_same_v<Outer, Diagonal>)
specfem::linear_system::Sum<1> operator*(const Outer &, const Inner &) {
  static_assert(specfem::linear_system_impl::always_false_v<Outer, Inner>,
                "specfem::linear_system: nested matrix products are not "
                "supported. Evaluate the inner product into a vector first.");
  return {};
}

/// Rejects two products in one expression, such as `A * x + B * y`
template <VectorProduct Left, VectorProduct Right>
specfem::linear_system::Sum<1> operator+(const Left &, const Right &) {
  static_assert(specfem::linear_system_impl::always_false_v<Left, Right>,
                "specfem::linear_system: an expression carries at most one "
                "matrix product. Accumulate the second one separately: "
                "x = A * u; x += B * v;");
  return {};
}

// ── Assignment ─────────────────────────────────────────────────────────────

inline VectorView &VectorView::operator=(const VectorView &other) {
  return *this = specfem::linear_system_impl::to_sum(other);
}

template <VectorExpression Expr>
VectorView &VectorView::operator=(const Expr &expr) {
  specfem::linear_system_impl::evaluate(*vector_, expr, true, *space_);
  return *this;
}

template <VectorExpression Expr>
VectorView &VectorView::operator+=(const Expr &expr) {
  specfem::linear_system_impl::evaluate(*vector_, expr, false, *space_);
  return *this;
}

inline VectorView &VectorView::operator+=(const VectorView &other) {
  return *this += specfem::linear_system_impl::to_sum(other);
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
template <VectorExpression Expr> type_real norm2(const Expr &expression) {
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
