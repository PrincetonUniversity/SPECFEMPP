#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/element.hpp"
#include "specfem/linear_system/sparse_matrix_view/mapping.hpp"
#include "specfem/linear_system/tpetra_types.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace specfem {
namespace linear_system {

/**
 * @brief A @ref Mapping whose global dof ids are Tpetra's global ordinal for
 * this build.
 *
 * @tparam DimensionTag Spatial dimension; must be `dim3`
 * @tparam MediumTag Medium whose degrees of freedom are numbered
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
using FEMapping = Mapping<DimensionTag, MediumTag,
                          specfem::linear_system::global_ordinal_type>;

/**
 * @brief The Tpetra objects of a finite-element linear system: the dof maps
 * and the sparsity graphs of the operators assembled over them.
 *
 * Built from a @ref Mapping, held by value. Two graphs are assembled once at
 * construction:
 *
 * - @ref full_matrix_graph, element-dense: the union over elements of one
 *   dense block each.
 * - @ref damping_matrix_graph, block-diagonal: the components at one mesh
 *   point coupled to each other and to nothing else.
 *
 * Serial-only: SPECFEM++ has no globally consistent point numbering across
 * ranks, so the constructor throws for a communicator with more than one rank
 * (follow-up of issue #1982).
 *
 * @tparam MappingType Dof numbering and connectivity; see @ref FEMapping
 */
template <typename MappingType> class FEAssembly {
public:
  /// Dof numbering and connectivity this system is built over
  using mapping_type = MappingType;

  /// Integer type of the global dof ids
  using global_ordinal_type = typename MappingType::global_ordinal_type;

  static_assert(
      std::is_same_v<global_ordinal_type,
                     specfem::linear_system::global_ordinal_type>,
      "FEAssembly requires a Mapping whose GlobalOrdinal is Tpetra's global "
      "ordinal for this build -- use specfem::linear_system::FEMapping.");

  /**
   * @brief Build the dof maps and the sparsity graphs of `mapping`.
   *
   * @param mapping Dof numbering and connectivity; copied in
   * @param comm Communicator the maps are defined on; must have exactly one
   *        rank
   *
   * @throws std::runtime_error if the communicator has more than one rank
   */
  explicit FEAssembly(MappingType mapping,
                      const Teuchos::RCP<const Teuchos::Comm<int>> &comm =
                          Tpetra::getDefaultComm())
      : mapping_(std::move(mapping)) {
    build_maps(comm);
    full_matrix_graph_ = build_full_matrix_graph();
    damping_matrix_graph_ = build_damping_matrix_graph();
  }

  /// Dof numbering and connectivity the graphs were built from
  inline const MappingType &mapping() const { return mapping_; }

  /// Uniquely-owned row map: contiguous `[0, num_global_dofs())`
  inline Teuchos::RCP<const map_type> owned_map() const { return owned_map_; }

  /**
   * @brief Fill-complete element-dense sparsity graph of the stiffness matrix,
   * on the owned map.
   *
   * @return Graph built at construction; never null
   */
  inline Teuchos::RCP<const fe_crs_graph_type> full_matrix_graph() const {
    return full_matrix_graph_;
  }

  /**
   * @brief Fill-complete block-diagonal sparsity graph of the damping matrix,
   * on the owned map.
   *
   * One dense `ncomp` block per absorbing-boundary point; interior rows are
   * empty, as is the whole graph on a mesh with no absorbing boundaries.
   *
   * @return Graph built at construction; never null
   */
  inline Teuchos::RCP<const fe_crs_graph_type> damping_matrix_graph() const {
    return damping_matrix_graph_;
  }

private:
  /// Field components per mesh point, from the mapping
  constexpr static int ncomponents = MappingType::ncomponents;

  /// Build the owned and owned+shared dof maps on `comm`
  void build_maps(const Teuchos::RCP<const Teuchos::Comm<int>> &comm) {
    comm_ = comm;
    if (comm_->getSize() > 1) {
      throw std::runtime_error(
          "specfem::linear_system::FEAssembly: distributed assembly on " +
          std::to_string(comm_->getSize()) +
          " ranks is not implemented yet. SPECFEM++ numbers global points "
          "per rank, and the cross-rank GID negotiation is a follow-up of "
          "issue #1982. Run on a single rank.");
    }

    const auto num_entries =
        static_cast<Tpetra::global_size_t>(mapping_.num_global_dofs());
    const global_ordinal_type index_base = 0;
    owned_map_ = Teuchos::rcp(new map_type(num_entries, index_base, comm_));
    // At one rank every dof is owned and nothing is shared.
    owned_plus_shared_map_ = owned_map_;
  }

  /**
   * @brief Assemble the element-dense sparsity graph.
   *
   * Two host passes over the element connectivity: the first bounds each row's
   * allocation exactly (`FECrsGraph` will not grow a row past its bound), the
   * second inserts. Duplicates at shared mesh points are merged by Tpetra.
   */
  Teuchos::RCP<const fe_crs_graph_type> build_full_matrix_graph() const {
    using device_type = fe_crs_graph_type::device_type;
    using dual_view_type = Kokkos::DualView<std::size_t *, device_type>;
    // What the FECrsGraph constructor takes; DualView has a converting copy
    // constructor from the non-const specialization, but no `const_type` alias.
    using const_dual_view_type =
        Kokkos::DualView<const std::size_t *, device_type>;

    const auto &elements = mapping_.elements();

    dual_view_type entries_per_row(
        "specfem::linear_system::FEAssembly::entries_per_row",
        static_cast<std::size_t>(mapping_.num_global_dofs()));

    // One buffer for the whole pass: a dof set is lazy, and
    // insertGlobalIndices below wants a contiguous array.
    const int ndof_e =
        ncomponents * mapping_.ngllz() * mapping_.nglly() * mapping_.ngllx();
    std::vector<global_ordinal_type> dofs(static_cast<std::size_t>(ndof_e));

    // Indexed by local row of owned_plus_shared_map_, which equals the global
    // id while assembly is single-rank (see the comm-size check in
    // build_maps).
    auto h_entries_per_row = entries_per_row.view_host();
    for (int i = 0; i < elements.size(); ++i) {
      mapping_(elements(i), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL)
          .expand(dofs.begin());
      for (const auto dof : dofs) {
        h_entries_per_row(static_cast<std::size_t>(dof)) += dofs.size();
      }
    }
    entries_per_row.modify_host();
    entries_per_row.sync_device();

    // Globally-indexed assembly: a dof set already yields global ids and
    // Tpetra builds the column map at endAssembly().
    auto graph = Teuchos::rcp(
        new fe_crs_graph_type(owned_map_, owned_plus_shared_map_,
                              const_dual_view_type(entries_per_row)));

    graph->beginAssembly();
    for (int i = 0; i < elements.size(); ++i) {
      mapping_(elements(i), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL)
          .expand(dofs.begin());
      for (const auto row : dofs) {
        graph->insertGlobalIndices(row, ndof_e, dofs.data());
      }
    }
    graph->endAssembly(); // migrates owned+shared -> owned, fill-completes

    return graph;
  }

  /**
   * @brief Assemble the block-diagonal damping sparsity graph.
   *
   * Same two passes as @ref build_full_matrix_graph, over damping points
   * rather than elements: `ncomp` entries per participating row, zero
   * everywhere else.
   */
  Teuchos::RCP<const fe_crs_graph_type> build_damping_matrix_graph() const {
    using device_type = fe_crs_graph_type::device_type;
    using dual_view_type = Kokkos::DualView<std::size_t *, device_type>;
    using const_dual_view_type =
        Kokkos::DualView<const std::size_t *, device_type>;

    const int nglob = mapping_.nglob();

    dual_view_type entries_per_row(
        "specfem::linear_system::FEAssembly::damping_entries_per_row",
        static_cast<std::size_t>(mapping_.num_global_dofs()));

    // Interior rows keep their zero bound: a row that takes no entries
    // allocates nothing.
    auto h_entries_per_row = entries_per_row.view_host();
    for (int point = 0; point < nglob; ++point) {
      if (!mapping_.is_damping_point(point)) {
        continue;
      }
      for (int icomp = 0; icomp < ncomponents; ++icomp) {
        h_entries_per_row(static_cast<std::size_t>(mapping_(point, icomp))) =
            ncomponents;
      }
    }
    entries_per_row.modify_host();
    entries_per_row.sync_device();

    auto graph = Teuchos::rcp(
        new fe_crs_graph_type(owned_map_, owned_plus_shared_map_,
                              const_dual_view_type(entries_per_row)));

    std::vector<global_ordinal_type> dofs(ncomponents);

    graph->beginAssembly();
    for (int point = 0; point < nglob; ++point) {
      if (!mapping_.is_damping_point(point)) {
        continue;
      }
      for (int icomp = 0; icomp < ncomponents; ++icomp) {
        dofs[icomp] = mapping_(point, icomp);
      }
      for (const auto row : dofs) {
        graph->insertGlobalIndices(row, ncomponents, dofs.data());
      }
    }
    graph->endAssembly(); // migrates owned+shared -> owned, fill-completes

    return graph;
  }

  MappingType mapping_; ///< Dof numbering and connectivity

  Teuchos::RCP<const Teuchos::Comm<int>> comm_;        ///< Communicator
  Teuchos::RCP<const map_type> owned_map_;             ///< Uniquely-owned rows
  Teuchos::RCP<const map_type> owned_plus_shared_map_; ///< Assembly rows

  Teuchos::RCP<const fe_crs_graph_type> full_matrix_graph_;    ///< K sparsity
  Teuchos::RCP<const fe_crs_graph_type> damping_matrix_graph_; ///< C sparsity
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
