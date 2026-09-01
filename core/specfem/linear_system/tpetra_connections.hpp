#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/tpetra_types.hpp"
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem {
namespace linear_system {

/**
 * @brief Tpetra side of the degree-of-freedom map: the communicator, the
 * row (owned) and column (overlap) maps, and sparsity-graph construction.
 *
 * Holds every Teuchos/Tpetra object of @ref DofMap, so the numbering half
 * (@ref DofNumbering) stays free of solver-library types. Swapping in a
 * different library that builds compressed-row matrices from a graph means
 * writing another class with this interface and composing it in
 * @ref BasicDofMap -- nothing in the numbering or in the assemblers has to
 * change.
 *
 * Serial-only in this milestone: SPECFEM++ numbers global points per rank
 * with no globally consistent numbering across ranks, so the constructor
 * throws for communicators with more than one rank. The owned and overlap
 * maps are kept separate in the API so distributed assembly (owned iglob ->
 * owned GIDs, shared-interface points -> overlap map, Export(ADD) into the
 * owned matrix) can be added without changing callers; at one rank they are
 * the same map.
 */
class TpetraConnections {
public:
  /**
   * @brief Signature of the visitor a sparsity pattern is replayed with.
   *
   * Called once per dense coupling block with the global dof ids of that
   * block. A pattern is any callable invocable as `pattern(visitor)`; it is
   * passed the visitor directly (not through this alias) so no type erasure
   * happens on the hot path. The alias documents the contract.
   */
  using BlockVisitor =
      std::function<void(std::span<const global_ordinal_type>)>;

  /// Empty connections: no communicator, no maps
  TpetraConnections() = default;

  /**
   * @brief Build the owned and overlap maps of `numbering` on `comm`.
   *
   * @tparam Numbering Dof numbering type; must provide `num_global_dofs()`
   * @param numbering Dof numbering the maps are built for
   * @param comm Teuchos communicator; must have exactly one rank
   */
  template <typename Numbering>
  TpetraConnections(const Numbering &numbering,
                    const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
      : comm_(comm) {
    if (comm_->getSize() > 1) {
      throw std::runtime_error(
          "specfem::linear_system::TpetraConnections: distributed assembly "
          "on " +
          std::to_string(comm_->getSize()) +
          " ranks is not implemented yet. SPECFEM++ numbers global points "
          "per rank, and the cross-rank GID negotiation is a follow-up of "
          "issue #1982. Run on a single rank.");
    }
    const Tpetra::global_size_t num_entries =
        static_cast<Tpetra::global_size_t>(numbering.num_global_dofs());
    const global_ordinal_type index_base = 0;
    owned_map_ = Teuchos::rcp(new map_type(num_entries, index_base, comm_));
    // At one rank every dof is owned; a distributed build replaces this with
    // a map that additionally holds the shared-interface dofs of neighbor
    // ranks.
    overlap_map_ = owned_map_;
  }

  /**
   * @brief Build a fill-complete sparsity graph from a coupling pattern.
   *
   * A pattern describes the matrix structure without naming any Tpetra type:
   * it is a callable invocable as `pattern(visitor)` that calls
   * `visitor(gids)` once per *dense coupling block* -- a set of global dof
   * ids that all couple to one another. Every dof of a spectral element
   * couples to every other dof of that element, so the stiffness pattern
   * emits one block per element; the Stacey damping operator is
   * block-diagonal, so it emits one `ncomp`-sized block per damping point.
   *
   * The pattern is replayed twice and must therefore be repeatable: once to
   * accumulate the per-row allocation bound (a row gets `block.size()`
   * entries for every block it appears in, which is exactly the number of
   * raw inserts the second pass makes -- duplicates are merged by
   * `fillComplete`), and once to insert the indices.
   *
   * @tparam Numbering Dof numbering type; must provide `num_global_dofs()`
   * @tparam Pattern Callable invocable as `pattern(visitor)`, where the
   *                 visitor matches @ref BlockVisitor
   * @param numbering Dof numbering the graph is built for
   * @param pattern Coupling blocks of the matrix
   * @return Fill-complete graph on the owned map
   */
  template <typename Numbering, typename Pattern>
  Teuchos::RCP<const crs_graph_type> build_graph(const Numbering &numbering,
                                                 const Pattern &pattern) const {
    // Indexed by global id: valid while assembly is single-rank (see the
    // comm-size check in the constructor), where local == global.
    std::vector<std::size_t> entries_per_row(
        static_cast<std::size_t>(numbering.num_global_dofs()), 0);

    pattern([&](std::span<const global_ordinal_type> gids) {
      for (const auto gid : gids) {
        entries_per_row[static_cast<std::size_t>(gid)] += gids.size();
      }
    });

    auto graph = Teuchos::rcp(new crs_graph_type(
        overlap_map_, Teuchos::ArrayView<const std::size_t>(
                          entries_per_row.data(), entries_per_row.size())));

    pattern([&](std::span<const global_ordinal_type> gids) {
      const auto num_columns = static_cast<int>(gids.size());
      for (const auto gid : gids) {
        graph->insertGlobalIndices(gid, num_columns, gids.data());
      }
    });

    graph->fillComplete(owned_map_, owned_map_);
    return graph;
  }

  /// Uniquely-owned row map: contiguous `[0, num_global_dofs())`
  inline Teuchos::RCP<const map_type> owned_map() const { return owned_map_; }

  /**
   * @brief Overlap (column) map: owned dofs plus, in a future distributed
   * build, the shared-interface dofs of neighbor ranks. Equal to
   * @ref owned_map at one rank.
   */
  inline Teuchos::RCP<const map_type> overlap_map() const {
    return overlap_map_;
  }

  /// Communicator the maps are defined on
  inline Teuchos::RCP<const Teuchos::Comm<int>> comm() const { return comm_; }

private:
  Teuchos::RCP<const Teuchos::Comm<int>> comm_; ///< Communicator
  Teuchos::RCP<const map_type> owned_map_;      ///< Uniquely-owned rows
  Teuchos::RCP<const map_type> overlap_map_;    ///< Owned + shared dofs
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
