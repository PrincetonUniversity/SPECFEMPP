#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/enums.hpp"
#include "specfem/linear_system/dof_map.hpp"
#include "specfem/linear_system/sparse_matrix_view/fe_assembly.hpp"
#include "specfem/linear_system/sparse_matrix_view/matrix_view.hpp"
#include <Teuchos_RCP.hpp>

namespace specfem {
namespace linear_system {

/**
 * @brief Assembles the Stacey damping matrix \f$ C \f$ of one medium as a
 * `Tpetra::CrsMatrix` by probing the velocity path of the production
 * stiffness kernel.
 *
 * The Stacey traction is pointwise in velocity, so \f$ C \f$ is
 * block-diagonal: one symmetric positive-semidefinite `ncomp x ncomp` block
 * per boundary GLL point,
 * \f$ C_p = \sum_{\mathrm{faces}} w J \left( \rho (v_p - v_s) \,
 * n \otimes n + \rho v_s \, I \right) \f$, accumulated over all faces
 * containing the point; interior points contribute exactly nothing.
 *
 * Probing runs the matrix-free `compute_stiffness_interaction` kernel --
 * which computes `accel += -K u - C v` -- with displacement \f$ \equiv 0 \f$
 * (the stiffness path contributes exact zeros) and unit velocity
 * \f$ v = e_c \f$ at every mesh point simultaneously; block-diagonality
 * means columns never mix, so `ncomp` kernel launches recover all blocks:
 * \f$ C(\cdot, c) = -\mathrm{accel} \f$. Probing the production kernel (not
 * a re-derivation of the traction) keeps the assembled \f$ C \f$ consistent
 * with the explicit solver by construction.
 *
 * The assembled operator satisfies \f$ C v = \f$ damping force
 * \f$ = -\mathrm{accel} \f$ of the matrix-free kernel at zero displacement
 * (before mass division), matching the equation of motion
 * \f$ M \ddot{u} + C \dot{u} + K u = f \f$.
 *
 * On meshes without Stacey boundaries the result is an empty (zero-entry)
 * matrix whose `apply()` is a no-op, so callers can use one code path.
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              dimension must be `dim3`; only `dim3, elastic, isotropic,
 *              none` is instantiated
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
class DampingAssembler {
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
   * @brief Validate scope and store the probe target.
   *
   * Throws `std::runtime_error` if any element of the medium is outside the
   * Stacey-tolerant stiffness scope (see @ref validate_stiffness_scope with
   * `StiffnessScope::with_stacey`) or if `dof_map` does not match the
   * assembly's forward field.
   *
   * @param assembly Assembled mesh, material properties, and fields; the
   *        forward field's displacement/velocity/acceleration storage is
   *        used as probe scratch and zeroed afterwards. Must outlive the
   *        assembler.
   * @param dof_map Dof numbering shared with the stiffness matrix (same
   *        instance the `StiffnessAssembler` exposes)
   */
  DampingAssembler(AssemblyType &assembly, const DofMap &dof_map);

  /**
   * @brief Assemble over a caller-supplied @ref FEAssembly.
   *
   * Same validation as the constructor above, but reuses `fe` -- its dof
   * numbering, its absorbing-boundary mask and its damping sparsity graph --
   * rather than building its own.
   *
   * @param assembly Assembled mesh, material properties, and fields; used as
   *        probe scratch and zeroed afterwards. Must outlive the assembler.
   * @param fe Dof maps and sparsity graphs of the medium; borrowed, and must
   *        outlive the assembler
   */
  DampingAssembler(AssemblyType &assembly, const FEAssemblyType &fe);

  /**
   * @brief Probe the velocity path and assemble the damping matrix.
   *
   * The matrix lives on @ref FEAssembly::damping_matrix_graph: a compact,
   * block-diagonal pattern with `ncomp` entries per row at damping (boundary)
   * points and empty rows at interior points. Every entry's `(row, column)`
   * pair also exists in the stiffness matrix's graph (same-point pairs are
   * same-element pairs), so the implicit Newmark operator can `sumInto`
   * \f$ C \f$ on the stiffness graph.
   *
   * All probe fields (displacement, velocity, acceleration) are zeroed on
   * host and device before returning.
   *
   * @return Fill-complete damping matrix on the owned map
   *
   * @throws std::runtime_error if the probe produces a nonzero block at a
   *         point the mesh does not tag as absorbing -- see
   *         @ref Mapping::is_damping_point
   */
  Teuchos::RCP<crs_matrix_type> assemble() const;

  /// Dof map shared by the matrix and any right-hand-side/solution vectors
  const DofMap &dof_map() const { return dof_map_; }

private:
  /// Validation shared by both constructors
  void validate() const;

  AssemblyType &assembly_; ///< Borrowed assembly (probe scratch, not owned)
  DofMap dof_map_;         ///< Per-medium dof numbering and maps

  /// Caller-supplied maps and graphs; null when this assembler builds its own
  const FEAssemblyType *fe_ = nullptr;
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
