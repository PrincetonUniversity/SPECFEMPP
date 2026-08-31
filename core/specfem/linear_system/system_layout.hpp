#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/linear_system/dof_map.hpp"
#include "specfem/setup.hpp"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <functional>
#include <vector>

namespace specfem {
namespace linear_system {

/**
 * @brief Maps the degrees of freedom of one medium onto Tpetra objects and
 * hands out the structural containers the assemblers fill.
 *
 * Owns the numbering (SPECFEM++ `(iglob, icomp)` \f$\rightarrow\f$ Tpetra
 * global id), the row (owned) and column (overlap) maps, and the sparsity
 * graphs. It deliberately owns *structure only*: the matrices it returns are
 * zero-valued and fill-complete-graph backed, so callers add values with
 * `sumIntoGlobalValues` and call `fillComplete()` themselves. Physics --
 * the stiffness probe, the Stacey velocity probe, the mass accumulation --
 * stays with the assemblers.
 *
 * The GID layout is component-blocked: `gid = icomp * nglob + iglob`, which
 * happens to line up with SPECFEM++ field storage
 * (`Kokkos::View<type_real **, Kokkos::LayoutLeft>` of shape
 * `(nglob, ncomp)`) and with the element-local ordering
 * `specfem::linear_system::local_dof_index`. Nothing depends on that
 * coincidence: @ref scatter and @ref gather address the field element by
 * element rather than exploiting contiguity, precisely so the ordering stays
 * free to change. The layout lives ONLY in `gid`, which is private -- change
 * it there to swap the ordering, and no consumer needs to follow.
 *
 * One instance describes one medium: `nglob` and `ncomp` are per-medium
 * quantities (a future multi-medium system holds one layout and one matrix
 * block per medium).
 *
 * Because both graphs come from one numbering, every `(row, column)` pair of
 * a @ref block_diagonal_matrix also exists in the @ref full_matrix graph --
 * same-point pairs are same-element pairs. The implicit Newmark operator
 * relies on this when it sums \f$ C \f$ onto \f$ K \f$'s graph; here it holds
 * by construction rather than by argument.
 *
 * Serial-only in this milestone: SPECFEM++ numbers global points per rank
 * with no globally consistent numbering across ranks, so the constructor
 * throws for communicators with more than one rank. The owned and overlap
 * maps are kept separate in the API so distributed assembly (owned iglob ->
 * owned GIDs, shared-interface points -> overlap map, Export(ADD) into the
 * owned matrix) can be added without changing callers; at one rank they are
 * the same map.
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              only `dim3, elastic, isotropic, none` is instantiated
 */
template <typename Tags> class SystemLayout {
public:
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto medium_tag = Tags::medium_tag;

  static_assert(dimension_tag == specfem::element::dimension_tag::dim3,
                "SystemLayout takes a dim3 assembly; Tags must be a dim3 "
                "bundle.");

  using AssemblyType = specfem::assembly::assembly<dimension_tag>;

  /**
   * @brief Host field storage of one medium, shaped `(nglob, ncomp)`.
   *
   * The type returned by `field_impl::get_host_field`, `get_host_field_dot`,
   * `get_host_field_dot_dot` and `get_host_mass_inverse`; also the shape
   * @ref scatter_point_block expects for one `ncomp x ncomp` block.
   */
  using host_field_view_type =
      Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  /**
   * @brief Construct the layout for the medium's degrees of freedom.
   *
   * Reads `nglob` from the forward simulation field and the component count
   * from the element attributes, then builds the owned and overlap maps.
   *
   * @param assembly Assembly with constructed fields; must outlive the layout
   * @param comm Teuchos communicator; must have exactly one rank
   */
  SystemLayout(const AssemblyType &assembly,
               const Teuchos::RCP<const Teuchos::Comm<int>> &comm);

  /**
   * @brief Build the layout of one medium from an assembled 3D simulation,
   * using `Tpetra::getDefaultComm()`.
   *
   * @param assembly Assembly with constructed fields; must outlive the layout
   * @return Layout for the medium
   */
  static SystemLayout from_assembly(const AssemblyType &assembly);

  /// Number of unique global mesh points of the medium
  inline int nglob() const { return nglob_; }

  /// Number of field components per mesh point
  inline int ncomp() const { return ncomp_; }

  /// Total number of degrees of freedom: `ncomp * nglob`
  inline global_ordinal_type num_global_dofs() const {
    return static_cast<global_ordinal_type>(ncomp_) * nglob_;
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

  /**
   * @brief Zero-valued matrix on the fully-connected element-connectivity
   * graph -- the structure of the stiffness matrix \f$ K \f$.
   *
   * Every dof of an element couples to every other dof of that element, so a
   * row interacts with all dofs of all elements sharing its mesh point. The
   * graph is built once and cached, so repeated calls are cheap and all
   * returned matrices share one graph.
   *
   * The matrix is backed by a fill-complete static graph: values start at
   * zero and only `sumIntoGlobalValues` is allowed. The caller fills it and
   * calls `fillComplete()`.
   *
   * @return Matrix on the cached full graph, ready to be summed into
   */
  Teuchos::RCP<crs_matrix_type> full_matrix() const;

  /**
   * @brief Zero-valued matrix on a block-diagonal graph -- one dense
   * `ncomp x ncomp` block per admitted mesh point, empty rows elsewhere.
   *
   * The structure of a pointwise operator such as the Stacey damping matrix
   * \f$ C \f$, whose blocks never couple different mesh points. A compact
   * graph matters here: a \f$ K \f$-sized graph would waste a full matrix of
   * memory on a block-diagonal operator.
   *
   * The graph is not cached -- it depends on `mask`.
   *
   * @param mask Predicate on `iglob`; only points for which it returns true
   *        get a block. An empty (default-constructed) function admits every
   *        point.
   * @return Matrix on the block-diagonal graph, ready to be summed into
   */
  Teuchos::RCP<crs_matrix_type>
  block_diagonal_matrix(const std::function<bool(int)> &mask = {}) const;

  /// Vector on the owned map, zero-initialized
  Teuchos::RCP<vector_type> create_vector() const;

  /**
   * @brief Copy a host field into an existing vector: field `(iglob, icomp)`
   * to row `gid(iglob, icomp)`.
   *
   * The in-place form exists so the time loop can reuse its state vectors
   * instead of allocating one per step.
   *
   * @param src Host field of shape `(nglob, ncomp)`
   * @param dst Vector on the owned map; every entry is overwritten
   */
  void scatter(const host_field_view_type &src, vector_type &dst) const;

  /**
   * @brief Allocating overload of @ref scatter, for callers that produce a
   * new vector rather than refilling one.
   *
   * @param src Host field of shape `(nglob, ncomp)`
   * @return Freshly allocated vector holding the scattered field
   */
  Teuchos::RCP<vector_type> scatter(const host_field_view_type &src) const;

  /**
   * @brief Copy a vector back into a host field -- the inverse of
   * @ref scatter.
   *
   * @param src Vector on the owned map
   * @param dst Host field of shape `(nglob, ncomp)`; every entry is
   *        overwritten
   */
  void gather(const vector_type &src, const host_field_view_type &dst) const;

  /**
   * @brief Add one `ncomp x ncomp` block at mesh point `iglob` into a matrix
   * carrying that point's block (see @ref block_diagonal_matrix).
   *
   * Throws `std::runtime_error` naming the point if the matrix's graph does
   * not carry the whole block -- use @ref has_point_block to ask without
   * throwing.
   *
   * @param matrix Target matrix, not yet fill-complete
   * @param iglob Mesh point whose block is being added
   * @param block Values of shape `(ncomp, ncomp)`, row-then-column
   */
  void scatter_point_block(crs_matrix_type &matrix, const int iglob,
                           const host_field_view_type &block) const;

  /**
   * @brief Whether `matrix` carries the full `ncomp x ncomp` block of mesh
   * point `iglob`.
   *
   * @param matrix Matrix to query
   * @param iglob Mesh point
   * @return True if every row and column of the point's block is present
   */
  bool has_point_block(const crs_matrix_type &matrix, const int iglob) const;

  /**
   * @brief Global column ids of one element in element-local dof order.
   *
   * Single source of truth for the ldof <-> gid correspondence used by both
   * the graph build and the element block scatter; entry `ldof` (see
   * @ref local_dof_index) holds `gid(iglob(ispec, iz, iy, ix), icomp)`.
   * Cached per element on first use.
   *
   * @param ispec Compute-domain element index
   * @return Column ids, indexed by element-local dof
   */
  const std::vector<global_ordinal_type> &
  element_column_gids(const int ispec) const;

private:
  /**
   * @brief Global id of component `icomp` at mesh point `iglob`.
   *
   * Component-blocked layout `gid = icomp * nglob + iglob` -- the single
   * source of truth for the global dof ordering. Private on purpose: every
   * consumer reaches row space through a named operation (@ref scatter,
   * @ref gather, @ref scatter_point_block, @ref element_column_gids), so
   * changing the ordering here changes nothing else.
   *
   * @param iglob Per-medium global point index in `[0, nglob())`
   * @param icomp Field component in `[0, ncomp())`
   * @return Tpetra global dof id in `[0, num_global_dofs())`
   */
  inline global_ordinal_type gid(const int iglob, const int icomp) const {
    return static_cast<global_ordinal_type>(icomp) * nglob_ + iglob;
  }

  /// Throw unless `field` has shape `(nglob, ncomp)`
  void validate_field_extents(const host_field_view_type &field,
                              const char *what) const;

  /// Build (or return the cached) fully-connected sparsity graph
  Teuchos::RCP<const crs_graph_type> full_graph() const;

  const AssemblyType *assembly_ = nullptr; ///< Borrowed assembly (not owned)
  int nglob_ = 0;                          ///< Points of the medium
  int ncomp_ = 0;                          ///< Components per point
  Teuchos::RCP<const Teuchos::Comm<int>> comm_; ///< Communicator
  Teuchos::RCP<const map_type> owned_map_;      ///< Uniquely-owned rows
  Teuchos::RCP<const map_type> overlap_map_;    ///< Owned + shared dofs

  // Caches. Mutable because the assemblers hold the layout by const
  // reference: building a graph does not change what the layout means, only
  // how often it has to be recomputed.
  mutable Teuchos::RCP<const crs_graph_type> full_graph_; ///< Cached K graph
  mutable std::vector<std::vector<global_ordinal_type>>
      element_columns_; ///< Cached per-element column ids
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
