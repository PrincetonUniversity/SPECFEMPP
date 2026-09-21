#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/operands.hpp"
#include "specfem/linear_system/tpetra_types.hpp"
#include "specfem/linear_system/vector_view/expression.hpp"
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCP.hpp>
#include <cstddef>
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

    ~Scratch() { space_.release(slot_); }

    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;

    /// The borrowed vector
    vector_type &vector() const { return *space_.scratch_[slot_]; }

  private:
    const VectorSpace &space_; ///< Pool the slot belongs to
    std::size_t slot_;         ///< Index into the pool
  };

  /// Borrow a scratch vector for the duration of one expression
  Scratch borrow() const {
    for (std::size_t slot = 0; slot < scratch_.size(); ++slot) {
      if (!in_use_[slot]) {
        in_use_[slot] = true;
        return Scratch(*this, slot);
      }
    }
    scratch_.push_back(Teuchos::rcp(new vector_type(map_)));
    in_use_.push_back(true);
    return Scratch(*this, scratch_.size() - 1);
  }

private:
  /// Return a borrowed slot to the pool
  void release(const std::size_t slot) const { in_use_[slot] = false; }

  Teuchos::RCP<const map_type> map_; ///< Row map of the space

  // Mutable so that an expression over const vectors can still borrow.
  mutable std::vector<Teuchos::RCP<vector_type>> scratch_; ///< Pool
  mutable std::vector<bool> in_use_; ///< Slot occupancy, by pool index
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

  /// The underlying Tpetra vector
  vector_type &vector() const { return *vector_; }

  /// The underlying Tpetra vector, for Belos and Ifpack2
  Teuchos::RCP<vector_type> rcp() const { return vector_; }

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
  template <typename Expression>
    requires(specfem::linear_system_impl::is_sum<Expression>::value ||
             specfem::linear_system_impl::is_product<Expression>::value ||
             specfem::linear_system_impl::is_expression<Expression>::value)
  VectorView &operator=(const Expression &expression);

  /// Accumulate an expression
  template <typename Expression>
    requires(specfem::linear_system_impl::is_sum<Expression>::value ||
             specfem::linear_system_impl::is_product<Expression>::value ||
             specfem::linear_system_impl::is_expression<Expression>::value)
  VectorView &operator+=(const Expression &expression);

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

/// Whether `T` can appear where a sum is expected
template <typename T>
constexpr bool is_sum_like_v =
    is_sum<T>::value ||
    std::is_same_v<std::remove_cvref_t<T>,
                   specfem::linear_system::VectorView> ||
    std::is_same_v<std::remove_cvref_t<T>, specfem::linear_system::vector_type>;

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
          const specfem::linear_system::Sum<N> &sum, bool &overwrite) {
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
    if (product.inner.terms[0].alpha == static_cast<scalar_type>(1)) {
      product.matrix->apply(*product.inner.terms[0].vector, target,
                            Teuchos::NO_TRANS, static_cast<scalar_type>(1),
                            beta);
      overwrite = false;
      return;
    }
  }

  const auto scratch = space.borrow();
  bool scratch_overwrite = true;
  emit(scratch.vector(), product.inner, scratch_overwrite);
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

  const auto scratch = space.borrow();
  bool scratch_overwrite = true;
  emit(scratch.vector(), product.inner, scratch_overwrite);
  target.elementWiseMultiply(static_cast<scalar_type>(1), *product.diagonal,
                             scratch.vector(), gamma);
  overwrite = false;
}

/// Write a sum-plus-product into `target`, plain part first
template <std::size_t N, typename Product>
void emit(specfem::linear_system::vector_type &target,
          const specfem::linear_system::Expression<N, Product> &expression,
          bool &overwrite, const specfem::linear_system::VectorSpace &space) {
  emit(target, expression.sum, overwrite);
  emit(target, expression.product, overwrite, space);
}

/**
 * @brief Evaluate any expression into `target`.
 *
 * Routes through scratch when the expression reads the target, so that
 * `x = 2 * x + y` is correct rather than order-dependent.
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

  if (overwrite && aliases(expression, &target)) {
    const auto scratch = space.borrow();
    bool scratch_overwrite = true;
    if constexpr (is_sum<Expr>::value) {
      emit(scratch.vector(), expression, scratch_overwrite);
    } else {
      emit(scratch.vector(), expression, scratch_overwrite, space);
    }
    if (scratch_overwrite) {
      target.putScalar(static_cast<scalar_type>(0));
    } else {
      target.update(static_cast<scalar_type>(1), scratch.vector(),
                    static_cast<scalar_type>(0));
    }
    return;
  }

  bool pending = overwrite;
  if constexpr (is_sum<Expr>::value) {
    emit(target, expression, pending);
  } else {
    emit(target, expression, pending, space);
  }
  if (pending) {
    target.putScalar(static_cast<scalar_type>(0));
  }
}

} // namespace linear_system_impl

namespace linear_system {

// ── Scaling and negation ───────────────────────────────────────────────────

/// Scale a vector or sum: `alpha * x`
template <typename Operand>
  requires(specfem::linear_system_impl::is_sum_like_v<Operand>)
constexpr auto operator*(const scalar_type alpha, const Operand &operand) {
  return specfem::linear_system_impl::scale(
      specfem::linear_system_impl::to_sum(operand), alpha);
}

/// Scale a vector or sum: `x * alpha`
template <typename Operand>
  requires(specfem::linear_system_impl::is_sum_like_v<Operand>)
constexpr auto operator*(const Operand &operand, const scalar_type alpha) {
  return alpha * operand;
}

/// Negate a vector or sum: `-x`
template <typename Operand>
  requires(specfem::linear_system_impl::is_sum_like_v<Operand>)
constexpr auto operator-(const Operand &operand) {
  return static_cast<scalar_type>(-1) * operand;
}

/// Negate a product: `-(A * x)`
template <typename Product>
  requires(specfem::linear_system_impl::is_product<Product>::value)
constexpr auto operator-(const Product &product) {
  return specfem::linear_system_impl::scale(product,
                                            static_cast<scalar_type>(-1));
}

// ── Sums ───────────────────────────────────────────────────────────────────

/// Add two vectors or sums
template <typename Left, typename Right>
  requires(specfem::linear_system_impl::is_sum_like_v<Left> &&
           specfem::linear_system_impl::is_sum_like_v<Right>)
constexpr auto operator+(const Left &left, const Right &right) {
  return specfem::linear_system_impl::concat(
      specfem::linear_system_impl::to_sum(left),
      specfem::linear_system_impl::to_sum(right));
}

/// Subtract two vectors or sums
template <typename Left, typename Right>
  requires(specfem::linear_system_impl::is_sum_like_v<Left> &&
           specfem::linear_system_impl::is_sum_like_v<Right>)
constexpr auto operator-(const Left &left, const Right &right) {
  return specfem::linear_system_impl::concat(
      specfem::linear_system_impl::to_sum(left),
      specfem::linear_system_impl::scale(
          specfem::linear_system_impl::to_sum(right),
          static_cast<scalar_type>(-1)));
}

// ── Products ───────────────────────────────────────────────────────────────

/// Apply a matrix: `A * x`
template <typename Operand>
  requires(specfem::linear_system_impl::is_sum_like_v<Operand>)
constexpr auto operator*(const crs_matrix_type &matrix,
                         const Operand &operand) {
  return specfem::linear_system::MatrixProduct<
      specfem::linear_system_impl::sum_type_t<Operand>>{
    &matrix, specfem::linear_system_impl::to_sum(operand)
  };
}

/// Apply a diagonal matrix: `diag(m) * x`
template <typename Operand>
  requires(specfem::linear_system_impl::is_sum_like_v<Operand>)
constexpr auto operator*(const Diagonal diagonal, const Operand &operand) {
  return specfem::linear_system::DiagProduct<
      specfem::linear_system_impl::sum_type_t<Operand>>{
    &diagonal.vector, specfem::linear_system_impl::to_sum(operand)
  };
}

// ── Sum plus product ───────────────────────────────────────────────────────

/// `x + A * y`
template <typename Left, typename Product>
  requires(specfem::linear_system_impl::is_sum_like_v<Left> &&
           specfem::linear_system_impl::is_product<Product>::value)
constexpr auto operator+(const Left &left, const Product &product) {
  return specfem::linear_system::Expression<
      specfem::linear_system_impl::sum_type_t<Left>::size, Product>{
    specfem::linear_system_impl::to_sum(left), product
  };
}

/// `A * y + x`
template <typename Product, typename Right>
  requires(specfem::linear_system_impl::is_product<Product>::value &&
           specfem::linear_system_impl::is_sum_like_v<Right>)
constexpr auto operator+(const Product &product, const Right &right) {
  return right + product;
}

/// `x - A * y`
template <typename Left, typename Product>
  requires(specfem::linear_system_impl::is_sum_like_v<Left> &&
           specfem::linear_system_impl::is_product<Product>::value)
constexpr auto operator-(const Left &left, const Product &product) {
  return left + specfem::linear_system_impl::scale(
                    product, static_cast<scalar_type>(-1));
}

/// `A * y - x`
template <typename Product, typename Right>
  requires(specfem::linear_system_impl::is_product<Product>::value &&
           specfem::linear_system_impl::is_sum_like_v<Right>)
constexpr auto operator-(const Product &product, const Right &right) {
  return specfem::linear_system_impl::scale(
             specfem::linear_system_impl::to_sum(right),
             static_cast<scalar_type>(-1)) +
         product;
}

// ── Rejected forms ─────────────────────────────────────────────────────────

/// Rejects `alpha * A * x`; scale the operand instead
template <typename Operand>
void operator*(const ScaledMatrix, const Operand &) {
  static_assert(
      !std::is_same_v<Operand, Operand>,
      "specfem::linear_system: a scaled matrix cannot be applied to a vector "
      "expression. Scale the operand instead -- A * (alpha * x) -- which is "
      "the same arithmetic in one fewer pass.");
}

/// Rejects `alpha * diag(m) * x`; scale the operand instead
template <typename Operand>
void operator*(const ScaledDiagonal, const Operand &) {
  static_assert(
      !std::is_same_v<Operand, Operand>,
      "specfem::linear_system: a scaled diagonal cannot be applied to a "
      "vector expression. Scale the operand instead: diag(m) * (alpha * x).");
}

/// Rejects a nested product such as `A * (B * x)`
template <typename Outer, typename Inner>
  requires(specfem::linear_system_impl::is_product<Inner>::value)
void operator*(const Outer &, const Inner &) {
  static_assert(!std::is_same_v<Inner, Inner>,
                "specfem::linear_system: nested matrix products are not "
                "supported. Evaluate the inner product into a vector first.");
}

/// Rejects two products in one expression, such as `A * x + B * y`
template <typename Left, typename Right>
  requires(specfem::linear_system_impl::is_product<Left>::value &&
           specfem::linear_system_impl::is_product<Right>::value)
void operator+(const Left &, const Right &) {
  static_assert(!std::is_same_v<Left, Left>,
                "specfem::linear_system: an expression carries at most one "
                "matrix product. Accumulate the second one separately: "
                "x = A * u; x += B * v;");
}

// ── Assignment ─────────────────────────────────────────────────────────────

inline VectorView &VectorView::operator=(const VectorView &other) {
  vector_->update(static_cast<scalar_type>(1), other.vector(),
                  static_cast<scalar_type>(0));
  return *this;
}

template <typename Expression>
  requires(specfem::linear_system_impl::is_sum<Expression>::value ||
           specfem::linear_system_impl::is_product<Expression>::value ||
           specfem::linear_system_impl::is_expression<Expression>::value)
VectorView &VectorView::operator=(const Expression &expression) {
  specfem::linear_system_impl::evaluate(*vector_, expression, true, *space_);
  return *this;
}

template <typename Expression>
  requires(specfem::linear_system_impl::is_sum<Expression>::value ||
           specfem::linear_system_impl::is_product<Expression>::value ||
           specfem::linear_system_impl::is_expression<Expression>::value)
VectorView &VectorView::operator+=(const Expression &expression) {
  specfem::linear_system_impl::evaluate(*vector_, expression, false, *space_);
  return *this;
}

inline VectorView &VectorView::operator+=(const VectorView &other) {
  vector_->update(static_cast<scalar_type>(1), other.vector(),
                  static_cast<scalar_type>(1));
  return *this;
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
template <typename Expression>
  requires(specfem::linear_system_impl::is_sum<Expression>::value ||
           specfem::linear_system_impl::is_product<Expression>::value ||
           specfem::linear_system_impl::is_expression<Expression>::value)
type_real norm2(const Expression &expression) {
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
