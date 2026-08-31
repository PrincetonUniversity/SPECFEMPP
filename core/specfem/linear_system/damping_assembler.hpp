#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/enums.hpp"
#include "specfem/linear_system/system_layout.hpp"
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
 *              only `dim3, elastic, isotropic, none` is instantiated
 */
template <typename Tags> class DampingAssembler {
public:
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto medium_tag = Tags::medium_tag;

  static_assert(dimension_tag == specfem::element::dimension_tag::dim3,
                "DampingAssembler takes a dim3 assembly; Tags must be a "
                "dim3 bundle.");

  /// Field components per mesh point of the medium
  constexpr static int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  using AssemblyType = specfem::assembly::assembly<dimension_tag>;
  using LayoutType = SystemLayout<Tags>;

  /**
   * @brief Validate scope and store the probe target.
   *
   * Throws `std::runtime_error` if any element of the medium is outside the
   * Stacey-tolerant stiffness scope (see @ref validate_stiffness_scope with
   * `StiffnessScope::with_stacey`).
   *
   * @param assembly Assembled mesh, material properties, and fields; the
   *        forward field's displacement/velocity/acceleration storage is
   *        used as probe scratch and zeroed afterwards. Must outlive the
   *        assembler.
   * @param layout Dof numbering shared with the stiffness matrix (same
   *        instance the `StiffnessAssembler` exposes)
   */
  DampingAssembler(AssemblyType &assembly, const LayoutType &layout);

  /**
   * @brief Probe the velocity path and assemble the damping matrix.
   *
   * The matrix comes from @ref SystemLayout::block_diagonal_matrix with the
   * damping-point mask: at most `ncomp` entries per row at damping
   * (boundary) points, empty rows at interior points. Because both graphs
   * come from one layout, every entry's `(row, column)` pair also exists in
   * the stiffness matrix's graph, so the implicit Newmark operator can
   * `sumInto` \f$ C \f$ on the stiffness graph.
   *
   * All probe fields (displacement, velocity, acceleration) are zeroed on
   * host and device before returning.
   *
   * @return Fill-complete damping matrix on the owned map
   */
  Teuchos::RCP<crs_matrix_type> assemble() const;

  /// Layout shared by the matrix and any right-hand-side/solution vectors
  const LayoutType &layout() const { return layout_; }

private:
  AssemblyType &assembly_; ///< Borrowed assembly (probe scratch, not owned)
  LayoutType layout_;      ///< Per-medium numbering, maps and graphs
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
