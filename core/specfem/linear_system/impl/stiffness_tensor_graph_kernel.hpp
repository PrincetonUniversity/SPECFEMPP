#pragma once

// Declares the TensorOperations-backed K_e producer. The whole file is guarded
// so that translation units may include it unconditionally; only the .cpp
// sibling (the sole TU that includes TensorOperations headers) exists in
// SPECFEM_ENABLE_TENSOROPS builds.
#ifdef SPECFEM_ENABLE_TENSOROPS

#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
template <specfem::element::dimension_tag DimensionTag> struct assembly;
} // namespace specfem::assembly

namespace specfem::linear_system_impl {

/**
 * @brief Compute dense element stiffness blocks \f$ K_e \f$ for a contiguous
 * batch of elements by evaluating the matrix-free action on every local unit
 * displacement through one declarative TensorOperations level graph.
 *
 * The graph is the production weak-form pipeline, stage for stage:
 * 1. **Gradient** -- reference-frame derivatives
 *    \f$ \partial u_c / \partial \xi_r \f$ as nine contractions of the staged
 *    displacement with the Lagrange derivative matrix
 *    (`specfem::algorithms::impl::element_gradient`'s summation).
 * 2. **Constitutive** -- one pointwise nine-output combine node whose functor
 *    loads the jacobian matrix and material properties at its global output
 *    coordinate, applies the chain rule, delegates to
 *    `specfem::medium_physics::compute_stress`, and returns the stress
 *    integrand \f$ F(c, r) = J \, \sigma_{cd} \, \partial r / \partial x_d \f$.
 *    This is the only medium-specific node: retargeting the kernel to another
 *    medium swaps this functor and the component arity.
 * 3. **Weighted divergence** -- nine contractions with the weighted transposed
 *    derivative matrix plus the transverse-weight combine
 *    (`specfem::algorithms::impl::element_divergence`'s summation).
 *
 * Assembly is the action evaluated on the identity: the staged input carries
 * one extra free column label over the \f$ n_{dof} = n_{comp} \cdot
 * N_{GLL}^3 \f$ local dofs, holding a unit displacement per column, and the
 * graph output IS \f$ K_e \f$ -- the same \f$ O(N_{GLL}^7) \f$ operation count
 * as the probe kernel
 * (@ref specfem::linear_system_impl::stiffness_probe_kernel, which stays the
 * correctness oracle), but expressed as batched regular contractions instead
 * of \f$ n_{dof} \f$ serialized probes with team barriers. A final
 * physics-free reshape writes the graph output into the `k_e`
 * (element, row, column) contract, owning the
 * @ref specfem::linear_system::local_dof_index ordering and the sign
 * convention (\f$ K u \f$ = internal force = \f$ -\mathrm{accel} \f$ before
 * mass division).
 *
 * The graph deliberately deviates from the `specfem::execution` iterator
 * idiom (`architecture/core-components/parallel-execution.md`):
 * TensorOperations owns its own team-level scheduling and scratch staging,
 * and piloting that library on a production path is the point of issue #2066.
 *
 * @tparam NGLL Number of GLL points per element edge (only 5 instantiated)
 * @tparam Tags Compile-time tags; dimension must be `dim3` and attenuation
 *              `none` (same scope as the probe kernel)
 * @param assembly Assembled mesh, jacobian matrix, and material properties
 * @param batch Contiguous element sub-range [begin, end) in compute-domain
 *              indices; the block for element `ispec` lands at slot
 *              `ispec - batch.begin_index()`
 * @param k_e Preallocated device buffer of shape `(>= batch size, ndof,
 *            ndof)` with `ndof = ncomp * NGLL^3`; same contract as
 *            @ref specfem::linear_system::compute_element_stiffness
 */
template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
void compute_element_stiffness_tensor_graph(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e);

} // namespace specfem::linear_system_impl

#endif // SPECFEM_ENABLE_TENSOROPS
