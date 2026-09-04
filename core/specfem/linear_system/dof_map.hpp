#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/linear_system/dof_numbering.hpp"
#include "specfem/linear_system/tpetra_connections.hpp"
#include "specfem/linear_system/tpetra_types.hpp"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>

namespace specfem {
namespace linear_system {

/**
 * @brief Maps SPECFEM++ (iglob, icomp) degrees of freedom of one medium to
 * solver global ids, and owns the row (owned) and column (overlap) maps.
 *
 * Composition of two independent halves, which is what keeps the linear
 * system portable across solver libraries:
 *
 * - `Numbering` (@ref DofNumbering) holds the SPECFEM++ quantities --
 *   `nglob`, `ncomp`, and the `gid` layout -- and names no library type.
 * - `Connections` (@ref TpetraConnections) holds everything the linear
 *   algebra library needs: the communicator, the maps, and sparsity-graph
 *   construction.
 *
 * Moving to another library that builds compressed-row matrices from a graph
 * is therefore a matter of writing a second `Connections` class and naming a
 * different alias; the numbering, the assemblers, and the solver are
 * untouched.
 *
 * The GID layout is component-blocked: `gid = icomp * nglob + iglob`,
 * deliberately matching SPECFEM++ field storage, so a solver vector maps 1:1
 * onto field memory with no permutation at solve time. It lives in
 * @ref DofNumbering::gid -- change it there to swap the ordering.
 *
 * One instance describes one medium: `nglob` and `ncomp` are per-medium
 * quantities (a future multi-medium system holds one map and one matrix
 * block per medium).
 *
 * @tparam Numbering Dof numbering half (see @ref DofNumbering)
 * @tparam Connections Linear-algebra-library half (see
 *                     @ref TpetraConnections)
 */
template <typename Numbering, typename Connections> class BasicDofMap {
public:
  /// Dof numbering half
  using numbering_type = Numbering;

  /// Linear-algebra-library half
  using connections_type = Connections;

  /// Integer type of the global dof ids
  using global_ordinal_type = typename Numbering::global_ordinal_type;

  /// Empty map: zero dofs, no communicator
  BasicDofMap() = default;

  /**
   * @brief Construct the map for `nglob` mesh points with `ncomp` components
   * per point.
   *
   * @param nglob Number of unique global mesh points of the medium
   * @param ncomp Number of field components per point (3 for 3D elastic)
   * @param comm Teuchos communicator; must have exactly one rank
   */
  BasicDofMap(const int nglob, const int ncomp,
              const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
      : numbering_(nglob, ncomp), connections_(numbering_, comm) {}

  /**
   * @brief Build the map of one medium from an assembled simulation.
   *
   * Reads `nglob` from the forward simulation field and the component count
   * from the element attributes, using `Tpetra::getDefaultComm()`.
   *
   * The tag bundle is a value argument rather than an explicit template
   * argument because a constructor cannot be given explicit template
   * arguments, and the medium is not deducible from `assembly`: an assembly
   * is templated on the dimension alone and holds the elements of every
   * medium, while one map describes one medium.
   *
   * @tparam Tags Compile-time tags (dimension, medium, property,
   *              attenuation); dimension must be `dim3`
   * @param assembly Assembly with constructed fields
   * @param tags Tag bundle selecting the medium; pass `Tags{}`
   */
  template <typename Tags>
    requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
  BasicDofMap(const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
              Tags tags)
      : numbering_(assembly, tags),
        connections_(numbering_, Tpetra::getDefaultComm()) {}

  /**
   * @brief Global id of component `icomp` at mesh point `iglob`.
   *
   * @param iglob Per-medium global point index in `[0, nglob())`
   * @param icomp Field component in `[0, ncomp())`
   * @return Global dof id in `[0, num_global_dofs())`
   */
  inline global_ordinal_type gid(const int iglob, const int icomp) const {
    return numbering_.gid(iglob, icomp);
  }

  /// Number of unique global mesh points of the medium
  inline int nglob() const { return numbering_.nglob(); }

  /// Number of field components per mesh point
  inline int ncomp() const { return numbering_.ncomp(); }

  /// Total number of degrees of freedom: `ncomp * nglob`
  inline global_ordinal_type num_global_dofs() const {
    return numbering_.num_global_dofs();
  }

  /// Uniquely-owned row map: contiguous `[0, num_global_dofs())`
  inline Teuchos::RCP<const map_type> owned_map() const {
    return connections_.owned_map();
  }

  /**
   * @brief Overlap (column) map: owned dofs plus, in a future distributed
   * build, the shared-interface dofs of neighbor ranks. Equal to
   * @ref owned_map at one rank.
   */
  inline Teuchos::RCP<const map_type> overlap_map() const {
    return connections_.overlap_map();
  }

  /// Communicator the maps are defined on
  inline Teuchos::RCP<const Teuchos::Comm<int>> comm() const {
    return connections_.comm();
  }

  /**
   * @brief Build a fill-complete sparsity graph from a coupling pattern.
   *
   * Forwards to the library half; see
   * @ref TpetraConnections::build_graph for what a pattern is and how it is
   * replayed. Callers describe *which dofs couple* and never touch a Tpetra
   * type to do so.
   *
   * @tparam Pattern Callable invocable as `pattern(visitor)`
   * @param pattern Coupling blocks of the matrix
   * @return Fill-complete graph on the owned map
   */
  template <typename Pattern>
  Teuchos::RCP<const crs_graph_type> build_graph(const Pattern &pattern) const {
    return connections_.build_graph(numbering_, pattern);
  }

  /// Dof numbering half
  inline const Numbering &numbering() const { return numbering_; }

  /// Linear-algebra-library half
  inline const Connections &connections() const { return connections_; }

private:
  // numbering_ is declared first: the connections are built from it.
  Numbering numbering_;     ///< SPECFEM++ dof numbering
  Connections connections_; ///< Maps and graphs of the solver library
};

/// Dof map of the Tpetra-backed linear system
using DofMap =
    BasicDofMap<DofNumbering<global_ordinal_type>, TpetraConnections>;

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
