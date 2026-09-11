#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/setup.hpp"
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

namespace specfem {
namespace linear_system {

/**
 * @brief Scalar used for assembled matrices and vectors.
 *
 * Aliased to `type_real` so the linear system follows the build's precision:
 * `float` by default (matching the float-only Tpetra installs on the
 * clusters), `double` with `SPECFEM_ENABLE_DOUBLE_PRECISION`. Single switch
 * point if the two ever need to diverge.
 */
using scalar_type = type_real;

/// Tpetra map with the default local/global ordinals and node type
using map_type = Tpetra::Map<>;

/// Global ordinal used for degree-of-freedom ids
using global_ordinal_type = map_type::global_ordinal_type;

/// Tpetra vector shared by mass, state, and right-hand-side vectors
using vector_type = Tpetra::Vector<scalar_type>;

/// Tpetra sparsity graph with the default ordinals and node type
using crs_graph_type = Tpetra::CrsGraph<>;

/// Assembled sparse matrix type (see @ref scalar_type for the precision)
using crs_matrix_type = Tpetra::CrsMatrix<scalar_type>;

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
