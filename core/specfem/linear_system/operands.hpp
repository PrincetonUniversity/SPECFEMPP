#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/tpetra_types.hpp"

namespace specfem {
namespace linear_system {

/**
 * @brief A matrix scaled by a coefficient, for `A += alpha * B`.
 *
 * Borrows its matrix; consume it in the same expression.
 */
struct ScaledMatrix {
  scalar_type alpha;             ///< Coefficient
  const crs_matrix_type &matrix; ///< Borrowed matrix
};

/**
 * @brief A diagonal matrix whose entries are a vector -- MATLAB's `diag(v)`.
 *
 * Lets a lumped mass vector be added as an operator: `A += c * diag(m)`.
 */
struct Diagonal {
  const vector_type &vector; ///< Borrowed diagonal entries
};

/// A @ref Diagonal scaled by a coefficient, for `A += alpha * diag(v)`
struct ScaledDiagonal {
  scalar_type alpha;         ///< Coefficient
  const vector_type &vector; ///< Borrowed diagonal entries
};

/**
 * @brief View a vector as the diagonal of a matrix.
 *
 * @param vector Diagonal entries; must outlive the expression
 * @return Wrapper accepted by `SparseMatrixView::operator+=`
 */
inline Diagonal diag(const vector_type &vector) { return Diagonal{ vector }; }

/**
 * @brief Scale a matrix for a pending sum.
 *
 * @param alpha Coefficient
 * @param matrix Matrix to scale; must outlive the expression
 * @return Wrapper accepted by `SparseMatrixView::operator+=`
 */
inline ScaledMatrix operator*(const scalar_type alpha,
                              const crs_matrix_type &matrix) {
  return ScaledMatrix{ alpha, matrix };
}

/**
 * @brief Scale a diagonal for a pending sum.
 *
 * @param alpha Coefficient
 * @param diagonal Diagonal to scale
 * @return Wrapper accepted by `SparseMatrixView::operator+=`
 */
inline ScaledDiagonal operator*(const scalar_type alpha,
                                const Diagonal diagonal) {
  return ScaledDiagonal{ alpha, diagonal.vector };
}

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
