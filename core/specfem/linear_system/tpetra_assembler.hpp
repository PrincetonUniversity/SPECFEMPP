#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/enums.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/linear_system/sparse_matrix_view/fe_assembly.hpp"
#include "specfem/linear_system/sparse_matrix_view/matrix_view.hpp"
#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>

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
 * materializes a global dense matrix: element blocks are computed on the
 * (Kokkos) device by the selected stiffness kernel, mirrored to the host,
 * and scattered into the sparse matrix with batched row updates. The
 * element batching that bounds the block buffer is an implementation
 * detail, not part of the API.
 *
 * One assembler produces the matrix block of one medium
 * (`Tags::medium_tag`); a future multi-medium system holds one assembler and
 * one matrix per medium. This milestone additionally requires the mesh to be
 * single-medium (fluid-solid coupling blocks are deferred) and serial (see
 * @ref FEAssembly).
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              dimension must be `dim3`; only `dim3, elastic, isotropic,
 *              none` is instantiated
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
class StiffnessAssembler {
public:
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto medium_tag = Tags::medium_tag;

  /// Field components per mesh point of the medium
  constexpr static int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  using AssemblyType = specfem::assembly::assembly<dimension_tag>;

  /// Dof numbering and connectivity of the medium
  using MappingType = FEMapping<dimension_tag, medium_tag>;

  /// Maps and sparsity graphs built over @ref MappingType
  using FEAssemblyType = FEAssembly<MappingType>;

  /**
   * @brief Validate scope and bind the dof maps and sparsity graphs.
   *
   * Throws `std::runtime_error` if any element is outside the supported
   * scope (see @ref validate_stiffness_scope) or if the mesh contains elements
   * of a medium other than `Tags::medium_tag` (single-medium milestone;
   * coupling blocks are deferred).
   *
   * @param assembly Assembled mesh, jacobian matrix, material properties,
   *        and fields; must outlive the assembler
   * @param fe Dof maps and sparsity graphs of the medium; borrowed, and must
   *        outlive the assembler
   * @param scope Boundary conditions the caller can represent (see
   *        @ref StiffnessScope); pass `with_stacey` only when the Stacey
   *        damping matrix is assembled separately
   * @param kernel_impl Kernel that fills the element blocks (see
   *        @ref StiffnessKernelImpl); the default follows the build
   *        (`direct` with TensorOperations, `probe` otherwise)
   */
  StiffnessAssembler(
      const AssemblyType &assembly, const FEAssemblyType &fe,
      const StiffnessScope scope = StiffnessScope::natural_boundaries,
      const StiffnessKernelImpl kernel_impl = default_stiffness_kernel_impl);

  /**
   * @brief Assemble the stiffness matrix.
   *
   * Fills the matrix through a @ref SparseMatrixView -- one block-diagonal
   * update per internal element batch -- on the element-dense graph of `fe`,
   * then closes it. Row/column ids follow @ref Mapping.
   *
   * @return Fill-complete stiffness matrix on the owned map
   */
  Teuchos::RCP<crs_matrix_type> assemble() const;

private:
  /**
   * @brief Elements whose stiffness blocks are formed per kernel launch.
   * Bounds the transient buffers: the `batch * ndof_e^2` block buffer plus
   * its host mirror (~36 MB each for 64 elastic NGLL = 5 elements in single
   * precision), and -- on the tensor-graph path -- that kernel's identity
   * and force workspaces, two more device views of the block buffer's
   * footprint each (~144 MB total device memory at the defaults). The direct
   * kernel allocates no workspace; its graph lives entirely in team scratch.
   */
  constexpr static int element_batch_size_ = 64;

  /// Fill element blocks in internal batches and scatter them into the matrix
  void fill_matrix(SparseMatrixView<MappingType> &matrix) const;

  const AssemblyType &assembly_;    ///< Borrowed assembly (not owned)
  const FEAssemblyType &fe_;        ///< Borrowed maps and sparsity graphs
  StiffnessKernelImpl kernel_impl_; ///< Element block producer
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
