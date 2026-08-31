#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/enums.hpp"
#include "specfem/linear_system/dof_map.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <vector>

namespace specfem {
namespace linear_system {

/**
 * @brief Assembles the global stiffness matrix \f$ K \f$ of one medium as a
 * `Tpetra::CrsMatrix` from dense element blocks.
 *
 * The assembled operator satisfies \f$ K u = \f$ internal force
 * \f$ = -\mathrm{accel} \f$ of the matrix-free
 * `compute_stiffness_interaction` kernel (before mass division) -- see
 * @ref compute_element_stiffness for the sign convention. Assembly never
 * materializes a global dense matrix: element blocks are computed in batches
 * on the (Kokkos) device by the stiffness probe kernel, mirrored to the
 * host, and scattered into the sparse matrix with batched row updates.
 *
 * One assembler produces the matrix block of one medium
 * (`Tags::medium_tag`); a future multi-medium system holds one assembler and
 * one matrix per medium. This milestone additionally requires the mesh to be
 * single-medium (fluid-solid coupling blocks are deferred) and serial (see
 * @ref DofMap).
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              only `dim3, elastic, isotropic, none` is instantiated
 */
template <typename Tags> class StiffnessAssembler {
public:
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto medium_tag = Tags::medium_tag;

  static_assert(dimension_tag == specfem::element::dimension_tag::dim3,
                "StiffnessAssembler takes a dim3 assembly; Tags must be a "
                "dim3 bundle.");

  /// Field components per mesh point of the medium
  constexpr static int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  /**
   * @brief Elements whose stiffness blocks are formed per probe-kernel
   * launch. Bounds the block buffer: `batch * ndof_e^2` scalars (~36 MB for
   * 64 elastic NGLL = 5 elements in single precision).
   */
  constexpr static int default_batch_size = 64;

  using AssemblyType = specfem::assembly::assembly<dimension_tag>;

  /**
   * @brief Validate scope and set up the dof map.
   *
   * Throws `std::runtime_error` if any element is outside the supported
   * scope (see @ref validate_stiffness_scope), if the mesh contains elements
   * of a medium other than `Tags::medium_tag` (single-medium milestone;
   * coupling blocks are deferred), or if the communicator has more than one
   * rank (see @ref DofMap).
   *
   * @param assembly Assembled mesh, jacobian matrix, material properties,
   *        and fields; must outlive the assembler
   * @param batch_size Elements per probe-kernel launch (>= 1)
   * @param scope Boundary conditions the caller can represent (see
   *        @ref StiffnessScope); pass `with_stacey` only when the Stacey
   *        damping matrix is assembled separately
   */
  StiffnessAssembler(
      const AssemblyType &assembly, const int batch_size = default_batch_size,
      const StiffnessScope scope = StiffnessScope::natural_boundaries);

  /**
   * @brief Assemble the stiffness matrix.
   *
   * Pipeline: build the `Tpetra::CrsGraph` from element connectivity (two
   * host passes over the index mapping), construct the matrix on the static
   * graph, fill it batch-by-batch with `sumIntoGlobalValues` row updates
   * from the probed element blocks, and `fillComplete()`. Row/column ids
   * follow @ref DofMap::gid.
   *
   * @return Fill-complete stiffness matrix on the owned map
   */
  Teuchos::RCP<crs_matrix_type> assemble() const;

  /// Dof map shared by the matrix and any right-hand-side/solution vectors
  const DofMap &dof_map() const { return dof_map_; }

private:
  /// Build and fill-complete the sparsity graph from element connectivity
  Teuchos::RCP<const crs_graph_type> build_graph() const;

  /// Probe element blocks in batches and scatter them into the matrix
  void fill_matrix(crs_matrix_type &matrix) const;

  /**
   * @brief Global column ids of one element in element-local dof order.
   *
   * Single source of truth for the ldof <-> gid correspondence used by both
   * the graph build and the block scatter; entry `ldof` (see
   * @ref local_dof_index) holds `dof_map_.gid(iglob(ispec, iz, iy, ix),
   * icomp)`.
   */
  std::vector<global_ordinal_type> element_column_gids(const int ispec) const;

  const AssemblyType &assembly_; ///< Borrowed assembly (not owned)
  DofMap dof_map_;               ///< Per-medium dof numbering and maps
  int batch_size_;               ///< Elements per probe-kernel launch
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
