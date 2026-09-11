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
 * batch of elements by direct sum-factored assembly of the weak form.
 *
 * Evaluates the closed form
 * \f[
 *   K_e[(c,i),(b,j)] = \sum_q \frac{\partial \phi_i^c}{\partial x_k}(q)\,
 *     C_{k c m b}\, \frac{\partial \phi_j^b}{\partial x_m}(q)\, w(q) J(q)
 * \f]
 * exploiting GLL collocation: \f$ \partial \phi_i / \partial \xi_r \f$ at
 * quadrature point \f$ q \f$ is nonzero only when \f$ i \f$ and \f$ q \f$
 * agree on the two coordinates transverse to \f$ r \f$, which collapses the
 * quadrature sum to a single point (cross reference-direction terms) or a 1D
 * line (diagonal terms). Cost is \f$ O(N_{GLL}^5) \f$ flops per element
 * against the probe kernel's \f$ O(N_{GLL}^7) \f$
 * (@ref specfem::linear_system_impl::stiffness_probe_kernel), which stays the
 * correctness oracle.
 *
 * Three stages:
 * 1. **Prologue** (plain Kokkos): per quadrature point, load the jacobian
 *    matrix and material properties through the same point-load API as the
 *    probe and fill the weighted material-metric tensor
 *    \f$ G_{rs}^{cb}(q) = w(q) J(q) [\lambda \xi_{r,c} \xi_{s,b}
 *      + \mu \xi_{r,b} \xi_{s,c} + \mu \delta_{cb} \sum_a \xi_{r,a}
 *      \xi_{s,a}] \f$.
 * 2. **Diagonal blocks** (TensorOperations LevelGraph, one execute per
 *    reference direction): contract \f$ G_{rr} \f$ with the Lagrange
 *    derivative outer product over the shared 1D quadrature line.
 * 3. **Epilogue** (plain Kokkos): per \f$ K_e \f$ entry, combine the three
 *    diagonal contractions with the six pointwise cross terms and write the
 *    block in @ref specfem::linear_system::local_dof_index ordering with the
 *    probe's sign convention (\f$ K u \f$ = internal force = \f$ -\mathrm{
 *    accel} \f$ before mass division).
 *
 * The middle stage deliberately deviates from the `specfem::execution`
 * iterator idiom (`architecture/core-components/parallel-execution.md`):
 * TensorOperations owns its own team-level scheduling, and piloting that
 * library on a production path is the point of issue #2066. The prologue and
 * epilogue have no per-element scratch or reduction structure, so they use
 * plain Kokkos range policies rather than chunked iterators.
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
void compute_element_stiffness_sum_factored(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e);

} // namespace specfem::linear_system_impl

#endif // SPECFEM_ENABLE_TENSOROPS
