#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/tpetra_types.hpp"

namespace specfem {
namespace linear_system {

template <typename MappingType> class SparseMatrixView;

/**
 * @brief A matrix scaled by a coefficient, for `A += alpha * B`.
 *
 * Built from a raw @ref crs_matrix_type or a @ref SparseMatrixView. Borrows
 * its matrix; consume it in the same expression.
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
 * @brief Scale a matrix view for a pending sum.
 *
 * @tparam MappingType Dof numbering of the view
 * @param alpha Coefficient
 * @param matrix Fill-complete view; must outlive the expression
 * @return Wrapper accepted by `SparseMatrixView::operator+=` and by the vector
 *         grammar's products
 */
template <typename MappingType>
ScaledMatrix operator*(const scalar_type alpha,
                       const SparseMatrixView<MappingType> &matrix) {
  return ScaledMatrix{ alpha, *matrix.matrix() };
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
