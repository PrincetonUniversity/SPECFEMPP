#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <stdexcept>
#include <string>

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

/**
 * @brief Maps SPECFEM++ (iglob, icomp) degrees of freedom of one medium to
 * Tpetra global ids and owns the row (owned) and column (overlap) maps.
 *
 * The GID layout is component-blocked: `gid = icomp * nglob + iglob`,
 * deliberately matching SPECFEM++ field storage
 * (`Kokkos::View<type_real **, Kokkos::LayoutLeft>` of shape
 * `(nglob, ncomp)`), so a Tpetra vector maps 1:1 onto field memory with no
 * permutation at solve time. It also matches the element-local ordering
 * `specfem::linear_system::local_dof_index`. The layout lives ONLY in
 * @ref gid -- change it there to swap the ordering.
 *
 * One DofMap instance describes one medium: `nglob` and `ncomp` are
 * per-medium quantities (a future multi-medium system holds one DofMap and
 * one matrix block per medium).
 *
 * Serial-only in this milestone: SPECFEM++ numbers global points per rank
 * with no globally consistent numbering across ranks, so the constructor
 * throws for communicators with more than one rank. The owned and overlap
 * maps are kept separate in the API so distributed assembly (owned iglob ->
 * owned GIDs, shared-interface points -> overlap map, Export(ADD) into the
 * owned matrix) can be added without changing callers; at one rank they are
 * the same map.
 */
class DofMap {
public:
  /**
   * @brief Construct the map for `nglob` mesh points with `ncomp` components
   * per point.
   *
   * @param nglob Number of unique global mesh points of the medium
   * @param ncomp Number of field components per point (3 for 3D elastic)
   * @param comm Teuchos communicator; must have exactly one rank
   */
  DofMap(const int nglob, const int ncomp,
         const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
      : nglob_(nglob), ncomp_(ncomp), comm_(comm) {
    if (comm_->getSize() > 1) {
      throw std::runtime_error(
          "specfem::linear_system::DofMap: distributed assembly on " +
          std::to_string(comm_->getSize()) +
          " ranks is not implemented yet. SPECFEM++ numbers global points "
          "per rank, and the cross-rank GID negotiation is a follow-up of "
          "issue #1982. Run on a single rank.");
    }
    const Tpetra::global_size_t num_entries =
        static_cast<Tpetra::global_size_t>(num_global_dofs());
    const global_ordinal_type index_base = 0;
    owned_map_ = Teuchos::rcp(new map_type(num_entries, index_base, comm_));
    // At one rank every dof is owned; a distributed build replaces this with
    // a map that additionally holds the shared-interface dofs of neighbor
    // ranks.
    overlap_map_ = owned_map_;
  }

  /**
   * @brief Build the DofMap of one medium from an assembled 3D simulation.
   *
   * Reads `nglob` from the forward simulation field and the component count
   * from the element attributes, using `Tpetra::getDefaultComm()`.
   *
   * @tparam MediumTag Medium whose degrees of freedom the map describes
   * @param assembly Assembly with constructed fields
   * @return DofMap for the medium
   */
  template <specfem::element::medium_tag MediumTag>
  static DofMap from_assembly(
      const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
          &assembly) {
    const auto &field =
        assembly.fields
            .get_simulation_field<specfem::simulation::field_type::forward>();
    const int nglob = field.get_nglob<MediumTag>();
    constexpr int ncomp =
        specfem::element::attributes<specfem::element::dimension_tag::dim3,
                                     MediumTag>::components;
    return DofMap(nglob, ncomp, Tpetra::getDefaultComm());
  }

  /**
   * @brief Global id of component `icomp` at mesh point `iglob`.
   *
   * Component-blocked layout `gid = icomp * nglob + iglob` -- the single
   * source of truth for the global dof ordering (see the class docs).
   *
   * @param iglob Per-medium global point index in `[0, nglob())`
   * @param icomp Field component in `[0, ncomp())`
   * @return Tpetra global dof id in `[0, num_global_dofs())`
   */
  inline global_ordinal_type gid(const int iglob, const int icomp) const {
    return static_cast<global_ordinal_type>(icomp) * nglob_ + iglob;
  }

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

  /// Communicator the maps are defined on
  inline Teuchos::RCP<const Teuchos::Comm<int>> comm() const { return comm_; }

private:
  int nglob_ = 0;                               ///< Points of the medium
  int ncomp_ = 0;                               ///< Components per point
  Teuchos::RCP<const Teuchos::Comm<int>> comm_; ///< Communicator
  Teuchos::RCP<const map_type> owned_map_;      ///< Uniquely-owned rows
  Teuchos::RCP<const map_type> overlap_map_;    ///< Owned + shared dofs
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
