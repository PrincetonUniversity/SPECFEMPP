#pragma once

#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <functional>

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
 * @brief Selects the kernel that fills the dense element stiffness blocks.
 *
 * `probe` applies the production matrix-free operator to every local unit
 * vector, one serialized probe at a time (correct by construction, the
 * reference implementation). `tensor_graph` evaluates the same action on all
 * unit columns at once through one declarative TensorOperations level graph
 * -- see @ref specfem::linear_system_impl::StiffnessTensorGraphKernel for
 * the pipeline; requesting it without `SPECFEM_ENABLE_TENSOROPS` throws
 * `std::runtime_error`. Both produce identical blocks up to roundoff (the
 * A/B test in `stiffness_tensor_graph_tests` holds them together).
 */
enum class StiffnessKernelImpl { probe, tensor_graph };

/**
 * @brief Default element stiffness kernel.
 *
 * `tensor_graph` when SPECFEM++ is built with TensorOperations -- enabling
 * the dependency is the opt-in -- and `probe` otherwise, so builds without
 * the flag are bit-identical to before the enum existed. Callers pin a
 * kernel explicitly (as the A/B test does) to override.
 */
#ifdef SPECFEM_ENABLE_TENSOROPS
inline constexpr StiffnessKernelImpl default_stiffness_kernel_impl =
    StiffnessKernelImpl::tensor_graph;
#else
inline constexpr StiffnessKernelImpl default_stiffness_kernel_impl =
    StiffnessKernelImpl::probe;
#endif

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
 * @param impl Kernel that fills the blocks (see @ref StiffnessKernelImpl);
 *             every implementation honors the contracts above
 */
template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void compute_element_stiffness(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e,
    const StiffnessKernelImpl impl = default_stiffness_kernel_impl);

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
 * @param impl Kernel that fills the blocks (see @ref StiffnessKernelImpl)
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void compute_element_stiffness(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e,
    const StiffnessKernelImpl impl = default_stiffness_kernel_impl);

/**
 * @brief Batched element-stiffness kernel bound to one assembly.
 *
 * Each call fills the leading `batch.size()` blocks of `k_e` under the
 * contract of @ref compute_element_stiffness, but may return before the
 * device work completes: a consumer reading `k_e` on the device must fence
 * first (a host mirror copy synchronizes by itself).
 */
using ElementStiffnessKernel =
    std::function<void(const specfem::datatype::ElementIndexRange &,
                       const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                          Kokkos::DefaultExecutionSpace> &)>;

/**
 * @brief Bind a stiffness kernel to an assembly for repeated batched calls.
 *
 * The one place a repeated caller (e.g. `StiffnessAssembler::fill_matrix`)
 * selects a kernel: per-construction costs are paid here once, not per
 * batch. For `tensor_graph` this constructs the workspace-owning
 * @ref specfem::linear_system_impl::StiffnessTensorGraphKernel (throwing
 * without `SPECFEM_ENABLE_TENSOROPS`); the probe kernel has no cross-batch
 * state and delegates to @ref compute_element_stiffness per call.
 *
 * Throws `std::runtime_error` for grids other than NGLL = 5 (the only 3D
 * instantiation).
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              dimension must be `dim3`
 * @param assembly Assembled mesh, jacobian matrix, and material properties;
 *        borrowed by the returned callable, and must outlive it
 * @param batch_capacity Largest batch a call may pass; sizes the
 *        tensor-graph workspace
 * @param impl Kernel that fills the blocks (see @ref StiffnessKernelImpl)
 * @return Callable filling `k_e` element blocks per contiguous batch
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
ElementStiffnessKernel make_element_stiffness_kernel(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const int batch_capacity,
    const StiffnessKernelImpl impl = default_stiffness_kernel_impl);

} // namespace linear_system
} // namespace specfem

namespace specfem::linear_system_impl {
/// Tag bundle for the only combination explicitly instantiated for the
/// linear system (issue #1982); shared by every instantiating TU.
using elastic_isotropic_tags =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
} // namespace specfem::linear_system_impl
