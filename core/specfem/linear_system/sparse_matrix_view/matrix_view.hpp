#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/sparse_matrix_view/dof_set.hpp"
#include "specfem/linear_system/tpetra_types.hpp"
#include <Kokkos_Core.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Access.hpp>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace specfem {
namespace linear_system {

/**
 * @brief A matrix scaled by a coefficient, for `A += alpha * B`.
 *
 * Produced by @ref operator*(scalar_type, const crs_matrix_type &); holds its
 * matrix by reference and is meant to be consumed in the same expression.
 */
struct ScaledMatrix {
  scalar_type alpha;             ///< Coefficient
  const crs_matrix_type &matrix; ///< Borrowed matrix
};

/**
 * @brief A diagonal matrix whose entries are a vector -- MATLAB's `diag(v)`.
 *
 * A lumped mass matrix is diagonal but stored as a `Tpetra::Vector`; wrapping
 * it lets `A += c * diag(m)` say what the operator is, instead of exposing the
 * storage choice through a `diagonal()` accessor on the matrix being built.
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

/**
 * @brief Writable view over a sparse matrix, indexed by @ref DofSet.
 *
 * Turns a scatter into the statement it actually is -- the finite-element
 * assembly line, with rows and columns named in mesh coordinates:
 *
 * @code
 * SparseMatrixView K(fe.full_matrix_graph(), fe.mapping());
 * K.begin_fill();
 * const auto edofs = fe.mapping()(batch, Kokkos::ALL, Kokkos::ALL,
 *                                 Kokkos::ALL, Kokkos::ALL);
 * K(edofs, edofs) += h_k_e;
 * K.finalize();
 * @endcode
 *
 * Rows and columns are always written out. A symmetric element block is two dof
 * sets that happen to be equal, which is both the honest description and the
 * form a rectangular block (a future fluid-solid coupling block) already fits
 * without a second grammar.
 *
 * The view owns the three things the raw Tpetra calls left to the caller:
 *
 * - **The graph invariant.** `sumIntoGlobalValues` silently drops entries
 *   outside the sparsity pattern and merely returns a count; @ref BlockRef
 *   checks that count on every row and throws naming the row.
 * - **Column expansion.** One scratch buffer for the whole assembly, rather
 *   than a `std::vector` per element.
 * - **Block layout.** Updates take a rank-2 or rank-3 `Kokkos::View`, so layout
 *   is a type rather than an assumption about a raw pointer.
 *
 * @tparam MappingType Dof numbering the matrix is built over; see
 *                     specfem::linear_system::FEMapping
 */
template <typename MappingType> class SparseMatrixView {
public:
  /// Integer type of the global dof ids
  using global_ordinal_type = typename MappingType::global_ordinal_type;

  /**
   * @brief How a block update combines with what the matrix already holds.
   */
  enum class UpdateMode {
    sum,    ///< Accumulate (`+=`); `sumIntoGlobalValues`
    replace ///< Overwrite (`=`); `replaceGlobalValues`
  };

  /**
   * @brief Pending update to one rectangular region of the matrix.
   *
   * Not constructed directly -- @ref SparseMatrixView::operator() returns one,
   * and it is consumed immediately by `+=` or `=`. It borrows the view and both
   * dof sets, so it must not outlive the full expression.
   *
   * @tparam Rows Row dof set
   * @tparam Cols Column dof set
   */
  template <typename Rows, typename Cols> class BlockRef {
  public:
    /**
     * @brief Bind a region of `view`.
     *
     * @param view Matrix being filled
     * @param rows Row dof set
     * @param cols Column dof set
     */
    BlockRef(SparseMatrixView &view, const Rows &rows, const Cols &cols)
        : view_(view), rows_(rows), cols_(cols) {}

    /**
     * @brief Accumulate a block into the matrix.
     *
     * @tparam Block Rank-2 or rank-3 `Kokkos::View`; see
     *         @ref SparseMatrixView::scatter for the shape rules
     * @param block Values to add
     */
    template <typename Block>
      requires(Kokkos::is_view<Block>::value)
    void operator+=(const Block &block) {
      view_.scatter(rows_, cols_, block, UpdateMode::sum);
    }

    /**
     * @brief Overwrite a block of the matrix.
     *
     * @tparam Block Rank-2 or rank-3 `Kokkos::View`; see
     *         @ref SparseMatrixView::scatter for the shape rules
     * @param block Values to write
     */
    template <typename Block>
      requires(Kokkos::is_view<Block>::value)
    void operator=(const Block &block) {
      view_.scatter(rows_, cols_, block, UpdateMode::replace);
    }

    /**
     * @brief Set every entry of the region to one value.
     *
     * @param value Value written to all `rows.size() * cols.size()` entries
     */
    void operator=(const scalar_type value) {
      view_.broadcast(rows_, cols_, value);
    }

  private:
    SparseMatrixView &view_; ///< Matrix being filled
    const Rows &rows_;       ///< Row dof set
    const Cols &cols_;       ///< Column dof set
  };

  /**
   * @brief Build a matrix on a sparsity graph.
   *
   * Values start at zero. The view is not fill-active until @ref begin_fill.
   *
   * @param graph Fill-complete sparsity pattern; typically
   *        specfem::linear_system::FEAssembly::full_matrix_graph
   * @param mapping Dof numbering the graph was built from; must outlive the
   *        view
   */
  SparseMatrixView(Teuchos::RCP<const fe_crs_graph_type> graph,
                   const MappingType &mapping)
      : mapping_(mapping),
        matrix_(Teuchos::rcp(new fe_crs_matrix_type(graph))) {}

  /// Dof numbering the matrix is built over
  const MappingType &mapping() const { return mapping_; }

  /**
   * @brief Open the matrix for value updates.
   *
   * On more than one rank this begins assembly on the owned+shared map; see
   * @ref finalize.
   */
  void begin_fill() {
    matrix_->beginAssembly();
    fill_active_ = true;
  }

  /**
   * @brief Close the matrix to value updates.
   *
   * Migrates owned+shared contributions to the owned map -- the `Export(ADD)`
   * that reproduces the matrix-free assembly sum across ranks -- and
   * fill-completes. A no-op migration at one rank.
   */
  void finalize() {
    matrix_->endAssembly();
    fill_active_ = false;
  }

  /// Whether the matrix currently accepts value updates
  bool is_fill_active() const { return fill_active_; }

  /// The assembled matrix; fill-complete once @ref finalize has run
  Teuchos::RCP<fe_crs_matrix_type> matrix() const { return matrix_; }

  /**
   * @brief Name a rectangular region of the matrix.
   *
   * @tparam Rows Row dof set
   * @tparam Cols Column dof set
   * @param rows Rows of the region
   * @param cols Columns of the region
   * @return Proxy to assign to or accumulate into
   *
   * @throws std::runtime_error if the view is not fill-active
   */
  template <typename Rows, typename Cols>
  BlockRef<Rows, Cols> operator()(const Rows &rows, const Cols &cols) {
    require_fill_active();
    return BlockRef<Rows, Cols>(*this, rows, cols);
  }

  /**
   * @brief Add another matrix: `A += B`.
   *
   * `B`'s sparsity must be contained in this matrix's, which is checked per
   * row rather than assumed.
   *
   * @param other Matrix to add; must be fill-complete
   */
  void operator+=(const crs_matrix_type &other) {
    add_matrix(static_cast<scalar_type>(1), other);
  }

  /**
   * @brief Add a scaled matrix: `A += alpha * B`.
   *
   * @param scaled Matrix and coefficient; see @ref ScaledMatrix
   */
  void operator+=(const ScaledMatrix scaled) {
    add_matrix(scaled.alpha, scaled.matrix);
  }

  /**
   * @brief Add a diagonal matrix: `A += diag(v)`.
   *
   * @param diagonal Diagonal entries; see @ref Diagonal
   */
  void operator+=(const Diagonal diagonal) {
    add_diagonal(static_cast<scalar_type>(1), diagonal.vector);
  }

  /**
   * @brief Add a scaled diagonal matrix: `A += alpha * diag(v)`.
   *
   * @param scaled Diagonal entries and coefficient; see @ref ScaledDiagonal
   */
  void operator+=(const ScaledDiagonal scaled) {
    add_diagonal(scaled.alpha, scaled.vector);
  }

private:
  /// What an update is being read from; only used to compose an error message
  enum class UpdateSource {
    block,   ///< A dense element block through @ref BlockRef
    matrix,  ///< Another matrix, through `operator+=`
    diagonal ///< A vector on the diagonal, through `operator+=`
  };

  /// Throw unless the matrix currently accepts value updates
  void require_fill_active() const {
    if (!fill_active_) {
      throw std::runtime_error(
          "specfem::linear_system::SparseMatrixView: the matrix is not "
          "fill-active; call begin_fill() before updating values.");
    }
  }

  /**
   * @brief Accumulate `alpha * other` into this matrix.
   *
   * One row at a time through @ref apply_row, so the containment of `other`'s
   * sparsity in this matrix's graph is enforced rather than assumed. Empty
   * rows are skipped, which is what makes adding an empty damping matrix free.
   *
   * @param alpha Coefficient applied to every entry
   * @param other Source matrix; must be fill-complete
   */
  void add_matrix(const scalar_type alpha, const crs_matrix_type &other) {
    require_fill_active();

    const auto num_rows = mapping_.num_global_dofs();
    const auto max_entries = other.getGlobalMaxNumRowEntries();
    typename crs_matrix_type::nonconst_global_inds_host_view_type columns(
        "specfem::linear_system::SparseMatrixView::add_matrix_columns",
        max_entries);
    typename crs_matrix_type::nonconst_values_host_view_type values(
        "specfem::linear_system::SparseMatrixView::add_matrix_values",
        max_entries);

    for (global_ordinal_type row = 0; row < num_rows; ++row) {
      std::size_t row_entries = 0;
      other.getGlobalRowCopy(row, columns, values, row_entries);
      if (row_entries == 0) {
        continue;
      }

      const auto count = static_cast<int>(row_entries);
      column_buffer_.resize(row_entries);
      value_buffer_.resize(row_entries);
      for (std::size_t k = 0; k < row_entries; ++k) {
        column_buffer_[k] = columns(k);
        value_buffer_[k] = alpha * values(k);
      }

      apply_row(row, value_buffer_.data(), count, UpdateMode::sum,
                UpdateSource::matrix);
    }
  }

  /**
   * @brief Accumulate `alpha * diag(vector)` into this matrix.
   *
   * Every diagonal entry is a self-pair, so it exists in any graph assembled
   * from coupling blocks; the per-row check still runs.
   *
   * @param alpha Coefficient applied to every entry
   * @param vector Diagonal entries, one per global dof
   */
  void add_diagonal(const scalar_type alpha, const vector_type &vector) {
    require_fill_active();

    const auto local = vector.getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto num_rows = mapping_.num_global_dofs();

    if (static_cast<global_ordinal_type>(local.extent(0)) != num_rows) {
      std::ostringstream message;
      message << "specfem::linear_system::SparseMatrixView: the matrix has "
              << num_rows << " rows but the diagonal holds " << local.extent(0)
              << " entries.";
      throw std::runtime_error(message.str());
    }

    column_buffer_.resize(1);
    for (global_ordinal_type row = 0; row < num_rows; ++row) {
      const scalar_type value = alpha * local(static_cast<std::size_t>(row), 0);
      column_buffer_[0] = row;
      apply_row(row, &value, 1, UpdateMode::sum, UpdateSource::diagonal);
    }
  }

  /**
   * @brief Scatter a dense or block-diagonal update into the matrix.
   *
   * The block's rank chooses the meaning, since a multi-element dof set is
   * ambiguous on its own:
   *
   * - **Rank 2** -- one dense block of exactly
   *   `rows.size() x cols.size()`.
   * - **Rank 3** -- `rows.outer_extent()` diagonal blocks, block `e` scattered
   *   to the dofs contributed by the `e`-th leading index. Each block is
   *   `rows.inner_size() x cols.inner_size()`.
   *
   * The dof sets are the authority on how much of the block is read, so a
   * rank-3 block may carry *more* leading blocks than the row set names: the
   * leading `outer_extent()` are used and the rest ignored. That is what admits
   * a reused, fixed-size batch scratch buffer, which is the dominant Kokkos
   * idiom and exactly how the stiffness probe allocates. Too few blocks throws.
   * The trailing extents must match exactly -- an oversized inner extent is not
   * a scratch buffer, it is a block from a different NGLL or component count,
   * and must not be silently truncated.
   *
   * @tparam Rows Row dof set
   * @tparam Cols Column dof set
   * @tparam Block Rank-2 or rank-3 `Kokkos::View`
   * @param rows Rows of the region
   * @param cols Columns of the region
   * @param block Values
   * @param mode Accumulate or overwrite
   *
   * @throws std::runtime_error on a shape mismatch, or if any entry of the
   *         block falls outside the sparsity graph
   */
  template <typename Rows, typename Cols, typename Block>
  void scatter(const Rows &rows, const Cols &cols, const Block &block,
               const UpdateMode mode) {
    constexpr auto rank = static_cast<int>(Block::rank());
    static_assert(rank == 2 || rank == 3,
                  "specfem::linear_system::SparseMatrixView: a block update "
                  "takes a rank-2 (one dense block) or rank-3 (one dense block "
                  "per leading index) view.");

    if constexpr (rank == 2) {
      require_extent(block.extent(0), rows.size(), "rows");
      require_extent(block.extent(1), cols.size(), "columns");

      column_buffer_.resize(cols.size());
      cols.expand(column_buffer_.begin());

      for (int r = 0; r < rows.size(); ++r) {
        scatter_row(rows[r], block, r, cols.size(), mode, rows.size());
      }
    } else {
      const int nblocks = rows.outer_extent();
      if (cols.outer_extent() != nblocks) {
        std::ostringstream message;
        message << "specfem::linear_system::SparseMatrixView: a rank-3 block "
                   "update needs the same number of leading indices in its row "
                   "and column sets, but the row set names "
                << nblocks << " and the column set names "
                << cols.outer_extent() << ".";
        throw std::runtime_error(message.str());
      }
      if (static_cast<int>(block.extent(0)) < nblocks) {
        std::ostringstream message;
        message << "specfem::linear_system::SparseMatrixView: the dof sets "
                   "name "
                << nblocks << " blocks but the update carries only "
                << block.extent(0) << ".";
        throw std::runtime_error(message.str());
      }
      require_extent(block.extent(1), rows.inner_size(), "rows");
      require_extent(block.extent(2), cols.inner_size(), "columns");

      const int nrows = rows.inner_size();
      const int ncols = cols.inner_size();
      column_buffer_.resize(ncols);

      for (int e = 0; e < nblocks; ++e) {
        cols.expand_block(e, column_buffer_.begin());
        const int base = e * nrows;
        for (int r = 0; r < nrows; ++r) {
          scatter_row(rows[base + r], block, e, r, ncols, mode, nrows);
        }
      }
    }
  }

  /**
   * @brief Write one value to every entry of a region.
   *
   * @tparam Rows Row dof set
   * @tparam Cols Column dof set
   * @param rows Rows of the region
   * @param cols Columns of the region
   * @param value Value written everywhere
   */
  template <typename Rows, typename Cols>
  void broadcast(const Rows &rows, const Cols &cols, const scalar_type value) {
    column_buffer_.resize(cols.size());
    cols.expand(column_buffer_.begin());
    value_buffer_.assign(cols.size(), value);

    for (int r = 0; r < rows.size(); ++r) {
      apply_row(rows[r], value_buffer_.data(), cols.size(), UpdateMode::replace,
                UpdateSource::block, rows.size());
    }
  }

  /// Scatter row `r` of a rank-2 block
  template <typename Block>
  void scatter_row(const global_ordinal_type row, const Block &block,
                   const int r, const int ncols, const UpdateMode mode,
                   const int nrows) {
    if constexpr (std::is_same_v<typename Block::array_layout,
                                 Kokkos::LayoutRight>) {
      // LayoutRight rows are contiguous, so the block's own storage is already
      // the array Tpetra wants.
      apply_row(row, &block(r, 0), ncols, mode, UpdateSource::block, nrows);
    } else {
      value_buffer_.resize(ncols);
      for (int c = 0; c < ncols; ++c) {
        value_buffer_[c] = block(r, c);
      }
      apply_row(row, value_buffer_.data(), ncols, mode, UpdateSource::block,
                nrows);
    }
  }

  /// Scatter row `r` of block `e` of a rank-3 block
  template <typename Block>
  void scatter_row(const global_ordinal_type row, const Block &block,
                   const int e, const int r, const int ncols,
                   const UpdateMode mode, const int nrows) {
    if constexpr (std::is_same_v<typename Block::array_layout,
                                 Kokkos::LayoutRight>) {
      apply_row(row, &block(e, r, 0), ncols, mode, UpdateSource::block, nrows);
    } else {
      value_buffer_.resize(ncols);
      for (int c = 0; c < ncols; ++c) {
        value_buffer_[c] = block(e, r, c);
      }
      apply_row(row, value_buffer_.data(), ncols, mode, UpdateSource::block,
                nrows);
    }
  }

  /**
   * @brief Apply one row update and verify it landed in full.
   *
   * Tpetra reports how many of the requested columns were in the graph and
   * silently drops the rest, so a short count means the matrix's sparsity
   * disagrees with the connectivity being assembled.
   *
   * @param row Global row id
   * @param values Row values, `ncols` of them
   * @param ncols Number of columns being updated
   * @param mode Accumulate or overwrite
   * @param source What the values came from, for the error message
   * @param nrows Rows in the block, for the error message; block sources only
   *
   * @throws std::runtime_error if any column fell outside the graph
   */
  void apply_row(const global_ordinal_type row, const scalar_type *values,
                 const int ncols, const UpdateMode mode,
                 const UpdateSource source, const int nrows = 0) {
    const int updated = (mode == UpdateMode::sum)
                            ? matrix_->sumIntoGlobalValues(
                                  row, ncols, values, column_buffer_.data())
                            : matrix_->replaceGlobalValues(
                                  row, ncols, values, column_buffer_.data());

    if (updated != ncols) {
      std::ostringstream message;
      message << "specfem::linear_system::SparseMatrixView: row " << row;
      switch (source) {
      case UpdateSource::block:
        message << " of a " << nrows << "x" << ncols << " block";
        break;
      case UpdateSource::matrix:
        message << ", adding another matrix";
        break;
      case UpdateSource::diagonal:
        message << ", adding a diagonal";
        break;
      }
      message << ": " << (ncols - updated) << " of " << ncols
              << " columns fell outside the graph. The matrix's sparsity does "
                 "not contain the values being added.";
      throw std::runtime_error(message.str());
    }
  }

  /// Throw unless a block extent matches what the dof sets require
  template <typename Extent>
  static void require_extent(const Extent actual, const int expected,
                             const std::string &what) {
    if (static_cast<int>(actual) != expected) {
      std::ostringstream message;
      message << "specfem::linear_system::SparseMatrixView: the dof sets name "
              << expected << " " << what << " but the update carries " << actual
              << ".";
      throw std::runtime_error(message.str());
    }
  }

  const MappingType &mapping_;              ///< Borrowed dof numbering
  Teuchos::RCP<fe_crs_matrix_type> matrix_; ///< Matrix being filled
  bool fill_active_{ false };               ///< Whether updates are allowed

  // Reused across every update of the assembly; grown on demand, never
  // per-element.
  std::vector<global_ordinal_type> column_buffer_; ///< Expanded column ids
  std::vector<scalar_type> value_buffer_;          ///< Row values, when the
                                                   ///< block is not LayoutRight
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
