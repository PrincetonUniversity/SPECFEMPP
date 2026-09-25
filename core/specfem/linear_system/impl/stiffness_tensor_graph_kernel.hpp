#pragma once

// Declares the TensorOperations-backed K_e producer, unconditionally: callers
// dispatch on it without preprocessor branches. The .cpp sibling is the sole
// TU that includes TensorOperations headers; without SPECFEM_ENABLE_TENSOROPS
// it defines these entry points as throwing stubs instead.

#include "specfem/datatype/element_index_range.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
template <specfem::element::dimension_tag DimensionTag> struct assembly;
} // namespace specfem::assembly

namespace specfem::linear_system_impl {

/**
 * @brief Batched producer of dense element stiffness blocks \f$ K_e \f$ that
 * evaluates the matrix-free action on every local unit displacement through
 * one declarative TensorOperations level graph, owning its device workspace
 * across batches.
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
 *    This is the medium-specific node (the nine-way arity around it is also
 *    3-component-specific): retargeting the kernel to another medium swaps
 *    this functor and the component arity.
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
 * The constructor allocates the workspace once at `batch_capacity` element
 * slots (the identity columns and force buffers -- the two dominant views,
 * each the footprint of the `k_e` block buffer), stages the two derivative
 * operators, and fills the identity; each call executes the graph over the
 * leading `batch.size()` slots. Construct once per assembly and call per
 * batch (see `StiffnessAssembler::fill_matrix`); the free wrapper
 * @ref compute_element_stiffness_tensor_graph constructs-and-calls for
 * one-shot use.
 *
 * The `k_e` device view is the producer/consumer seam for the planned
 * device-side CRS scatter follow-up: a consumer that scatters element blocks
 * into the sparse matrix on device replaces today's host mirror without
 * touching this class.
 *
 * The graph deliberately deviates from the `specfem::execution` iterator
 * idiom (`architecture/core-components/parallel-execution.md`):
 * TensorOperations owns its own team-level scheduling and scratch staging,
 * and piloting that library on a production path is the point of issue #2066.
 *
 * Without SPECFEM_ENABLE_TENSOROPS every member is a stub throwing
 * `std::runtime_error`.
 *
 * @tparam NGLL Number of GLL points per element edge (only 5 instantiated)
 * @tparam Tags Compile-time tags; dimension must be `dim3` and attenuation
 *              `none` (same scope as the probe kernel)
 */
template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
class StiffnessTensorGraphKernel {
public:
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto medium_tag = Tags::medium_tag;

  /// Field components per mesh point of the medium
  constexpr static int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  /// Dense block edge length: rows/columns of \f$ K_e \f$
  constexpr static int ndof = ncomp * NGLL * NGLL * NGLL;

  using AssemblyType = specfem::assembly::assembly<dimension_tag>;
  using StiffnessViewType = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                         Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Allocate and fill the graph workspace for up to `batch_capacity`
   * elements.
   *
   * Throws `std::runtime_error` if the mesh grid does not match `NGLL` or
   * `batch_capacity` is not positive.
   *
   * @param assembly Assembled mesh, jacobian matrix, and material properties;
   *        borrowed, must outlive the kernel
   * @param batch_capacity Largest batch a call may pass
   */
  StiffnessTensorGraphKernel(const AssemblyType &assembly, int batch_capacity);

  /**
   * @brief Fill the leading `batch.size()` blocks of `k_e`.
   *
   * Does not fence: the mirror copy a host-scattering caller issues next
   * synchronizes; a caller that consumes `k_e` on device must fence first.
   *
   * @param batch Contiguous element sub-range [begin, end) in compute-domain
   *        indices, at most `batch_capacity` long; the block for element
   *        `ispec` lands at slot `ispec - batch.begin_index()`
   * @param k_e Preallocated device buffer of shape `(>= batch size, ndof,
   *            ndof)`; same contract as
   *            @ref specfem::linear_system::compute_element_stiffness
   */
  void operator()(const specfem::datatype::ElementIndexRange &batch,
                  const StiffnessViewType &k_e) const;

private:
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  /// Staged 1D derivative operator (point, function) or its weighted
  /// transpose
  using OperatorViewType =
      Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, ExecSpace>;

  /// Graph-shaped buffer (component, element slot, column, iz, iy, ix)
  using WorkspaceViewType = Kokkos::View<type_real ***[NGLL][NGLL][NGLL],
                                         Kokkos::LayoutRight, ExecSpace>;

  const AssemblyType &assembly_; ///< Borrowed assembly (not owned)
  int batch_capacity_;           ///< Element slots the workspace holds

  OperatorViewType derivative_;         ///< hprime(point, function)
  OperatorViewType weighted_transpose_; ///< hprime(summed, point) * w(summed)
  WorkspaceViewType unit_columns_;      ///< Identity input, filled once
  WorkspaceViewType forces_;            ///< Graph output before the reshape
};

/**
 * @brief One-shot wrapper over @ref StiffnessTensorGraphKernel:
 * construct-and-call sized to `batch.size()`.
 *
 * Repeated batched use should construct the kernel once instead (the
 * workspace allocation and identity fill are per-construction costs).
 * Parameters and the `k_e` contract match the class call operator; fences
 * before returning, matching @ref
 * specfem::linear_system::compute_element_stiffness.
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
