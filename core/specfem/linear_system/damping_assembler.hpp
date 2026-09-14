#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/enums.hpp"
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
 * which computes `accel += -K u - C v` -- at displacement
 * \f$ \equiv 0 \f$ and unit velocity \f$ v = e_c \f$ at every mesh point at
 * once. Block-diagonality means columns never mix, so `ncomp` launches
 * recover every block: \f$ C(\cdot, c) = -\mathrm{accel} \f$.
 *
 * The result satisfies \f$ M \ddot{u} + C \dot{u} + K u = f \f$ against the
 * matrix-free kernel, before mass division. On meshes without Stacey
 * boundaries it is an empty matrix whose `apply()` is a no-op.
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
   * `StiffnessScope::with_stacey`) or if `fe` does not match the assembly's
   * forward field.
   *
   * @param assembly Assembled mesh, material properties, and fields; the
   *        forward field's displacement/velocity/acceleration storage is
   *        used as probe scratch and zeroed afterwards. Must outlive the
   *        assembler.
   * @param fe Dof maps, the absorbing-boundary mask and the damping sparsity
   *        graph of the medium; borrowed, and must outlive the assembler
   */
  DampingAssembler(AssemblyType &assembly, const FEAssemblyType &fe);

  /**
   * @brief Probe the velocity path and assemble the damping matrix.
   *
   * The matrix lives on @ref FEAssembly::damping_matrix_graph: `ncomp`
   * entries per row at damping (boundary) points, empty rows at interior
   * points. Every entry's `(row, column)` pair also exists in the stiffness
   * graph, so the implicit Newmark operator can `sumInto` \f$ C \f$ there.
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

private:
  AssemblyType &assembly_;   ///< Borrowed assembly (probe scratch, not owned)
  const FEAssemblyType &fe_; ///< Borrowed maps, mask and sparsity graph
};

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
