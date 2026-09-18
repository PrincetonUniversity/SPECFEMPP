#pragma once

#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
template <specfem::element::dimension_tag DimensionTag> struct assembly;
} // namespace specfem::assembly

namespace specfem {
namespace linear_system {

/**
 * @brief Local degree-of-freedom index within one element.
 *
 * Component-blocked ordering, deliberately matching the global DOF layout
 * `gid = icomp * nglob + iglob` used by the linear system:
 * \f$ \mathrm{ldof} = \mathrm{icomp} \cdot \mathrm{NGLL}^3 +
 * (iz \cdot \mathrm{NGLL} + iy) \cdot \mathrm{NGLL} + ix \f$.
 *
 * This is the single source of truth for the row/column ordering of the
 * element stiffness blocks produced by @ref compute_element_stiffness.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @param icomp Field component (ranges over the medium's component count)
 * @param iz GLL index in the z direction
 * @param iy GLL index in the y direction
 * @param ix GLL index in the x direction
 * @return Local dof index in `[0, ncomp * NGLL^3)`
 */
template <int NGLL>
KOKKOS_INLINE_FUNCTION constexpr int
local_dof_index(const int icomp, const int iz, const int iy, const int ix) {
  return icomp * NGLL * NGLL * NGLL + (iz * NGLL + iy) * NGLL + ix;
}

/**
 * @brief Boundary conditions the caller's probe/assembly can represent.
 *
 * `natural_boundaries` keeps the historical strict check: only `none` and
 * `acoustic_free_surface` boundary tags (natural boundary conditions) are
 * admitted. `with_stacey` additionally admits `boundary_tag::stacey` -- valid
 * because the displacement probe runs with velocity \f$ \equiv 0 \f$, where
 * the Stacey dashpot contributes exactly nothing to \f$ K \f$; the caller
 * must assemble the damping matrix \f$ C \f$ separately (see
 * @ref DampingAssembler). `composite_stacey_dirichlet` stays rejected in
 * both scopes (a Dirichlet mask is not representable yet).
 */
enum class StiffnessScope { natural_boundaries, with_stacey };

/**
 * @brief Verify that every element of `Tags::medium_tag` is within the scope
 * supported by the stiffness probe.
 *
 * Throws `std::runtime_error` naming the offending element and tag unless all
 * elements of the medium match `Tags::property_tag`, have
 * `attenuation_tag::none`, a boundary tag admitted by `scope` (see
 * @ref StiffnessScope), and the mesh uses NGLL = 5 (the only 3D
 * instantiation).
 *
 * The check is per-medium: a mixed mesh passes as long as the elements of
 * `Tags::medium_tag` conform. Restrictions on the mesh as a whole (e.g.
 * single-medium, deferring fluid-solid coupling) belong to the caller.
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              dimension must be `dim3`
 * @param assembly Assembled mesh, element types, and material properties
 * @param scope Boundary conditions the caller can represent; defaults to the
 *              strict natural-boundaries check
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void validate_stiffness_scope(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const StiffnessScope scope = StiffnessScope::natural_boundaries);

/**
 * @brief Compute dense element stiffness blocks \f$ K_e \f$ for a contiguous
 * batch of elements by probing the single-element operator with local unit
 * vectors.
 *
 * Each probe applies the matrix-free element operator
 * (gradient \f$\rightarrow\f$ stress \f$\rightarrow\f$ divergence) to a unit
 * displacement at one local dof, with velocity \f$ \equiv 0 \f$ (pure
 * \f$ K \f$), no mass-matrix division, and no boundary terms.
 *
 * Sign convention: \f$ K_e(i,j) \f$ is the divergence result at row dof
 * \f$ i \f$ when probing unit column dof \f$ j \f$, so \f$ K u \f$ equals the
 * internal force. The matrix-free time-marching kernel accumulates
 * `accel += -(divergence result)`, hence \f$ K u = -\mathrm{accel} \f$
 * (before mass division). Row/column ordering follows @ref local_dof_index.
 *
 * The kernel runs in `Kokkos::DefaultExecutionSpace`; the same code path
 * serves CPU and GPU builds.
 *
 * @tparam NGLL Number of GLL points per element edge (only 5 instantiated)
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              dimension must be `dim3` and `Tags::attenuation_tag` must be
 *              `none`
 * @param assembly Assembled mesh, jacobian matrix, and material properties
 * @param batch Contiguous element sub-range [begin, end) in compute-domain
 *              indices; the block for element `ispec` lands at slot
 *              `ispec - batch.begin_index()`
 * @param k_e Preallocated device buffer of shape `(batch_elements, ndof_e,
 *            ndof_e)` with `extent(0) >= batch.size()` and
 *            `extent(1) == extent(2) == ndof_e = ncomp * NGLL^3` (375 for 3D
 *            elastic with NGLL = 5). LayoutRight keeps each block row
 *            contiguous on the host mirror so rows can be handed directly to
 *            batched sparse-matrix row inserts.
 */
template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void compute_element_stiffness(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e);

/**
 * @brief Runtime NGLL dispatcher for @ref compute_element_stiffness.
 *
 * Reads `assembly.mesh.element_grid` and forwards to the matching NGLL
 * instantiation. Throws `std::runtime_error` for grids other than 5 (the
 * only 3D instantiation, mirroring the time-marching solver).
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              dimension must be `dim3`
 * @param assembly Assembled mesh, jacobian matrix, and material properties
 * @param batch Contiguous element sub-range [begin, end)
 * @param k_e Preallocated device buffer (see the NGLL overload)
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void compute_element_stiffness(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e);

} // namespace linear_system
} // namespace specfem
