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
 * materializes a global dense matrix: element blocks are computed in batches
 * on the (Kokkos) device by the stiffness probe kernel, mirrored to the
 * host, and scattered into the sparse matrix with batched row updates.
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

  /**
   * @brief Elements whose stiffness blocks are formed per probe-kernel
   * launch. Bounds the block buffer: `batch * ndof_e^2` scalars (~36 MB for
   * 64 elastic NGLL = 5 elements in single precision).
   */
  constexpr static int default_batch_size = 64;

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
   * `fe` is supplied rather than built here so that several operators over the
   * same mesh share one description of it -- each sparsity graph costs two host
   * passes over the connectivity, and there is no reason to pay for them per
   * assembler.
   *
   * @param assembly Assembled mesh, jacobian matrix, material properties,
   *        and fields; must outlive the assembler
   * @param fe Dof maps and sparsity graphs of the medium; borrowed, and must
   *        outlive the assembler
   * @param batch_size Elements per probe-kernel launch (>= 1)
   * @param scope Boundary conditions the caller can represent (see
   *        @ref StiffnessScope); pass `with_stacey` only when the Stacey
   *        damping matrix is assembled separately
   */
  StiffnessAssembler(
      const AssemblyType &assembly, const FEAssemblyType &fe,
      const int batch_size = default_batch_size,
      const StiffnessScope scope = StiffnessScope::natural_boundaries);

  /**
   * @brief Assemble the stiffness matrix.
   *
   * Fills the matrix batch-by-batch through a @ref SparseMatrixView -- one
   * block-diagonal update per probe batch -- on the element-dense graph of
   * `fe`, then closes it. Row/column ids follow @ref Mapping.
   *
   * @return Fill-complete stiffness matrix on the owned map
   */
  Teuchos::RCP<crs_matrix_type> assemble() const;

private:
  /// Probe element blocks in batches and scatter them into the matrix
  void fill_matrix(SparseMatrixView<MappingType> &matrix) const;

  const AssemblyType &assembly_; ///< Borrowed assembly (not owned)
  const FEAssemblyType &fe_;     ///< Borrowed maps and sparsity graphs
  int batch_size_;               ///< Elements per probe-kernel launch
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
