#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/setup.hpp"
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_FECrsGraph.hpp>
#include <Tpetra_FECrsMatrix.hpp>
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

/**
 * @brief Finite-element sparsity graph: assembled over an owned+shared dof map
 * and migrated to an owned map by `endAssembly()`.
 *
 * Derives from @ref crs_graph_type, so an `fe_crs_graph_type` is accepted
 * anywhere a `crs_graph_type` is.
 */
using fe_crs_graph_type = Tpetra::FECrsGraph<>;

/**
 * @brief Finite-element matrix built on an @ref fe_crs_graph_type.
 *
 * `beginAssembly()`/`endAssembly()` bracket the value fill and perform the
 * owned+shared to owned migration. Derives from @ref crs_matrix_type.
 */
using fe_crs_matrix_type = Tpetra::FECrsMatrix<scalar_type>;

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
