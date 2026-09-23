#pragma once

// Declares the direct (sum-factored) TensorOperations K_e producer,
// unconditionally: callers dispatch on it without preprocessor branches. The
// .cpp sibling includes TensorOperations headers; without
// SPECFEM_ENABLE_TENSOROPS it defines these entry points as throwing stubs.

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
 * @brief Producer of dense element stiffness blocks \f$ K_e \f$ that writes
 * every entry in closed form through one TensorOperations level graph of two
 * contractions (`TensorOperations::make_einsum_node`: the labels decide what
 * is summed).
 *
 * The block is the bilinear form
 * \f[
 *   K_e(a, i; b, j) = \sum_q \frac{\partial \phi_i}{\partial x_c}(q)\,
 *   C_{a c b d}(q)\, \frac{\partial \phi_j}{\partial x_d}(q)\, w(q) J(q),
 *   \qquad
 *   \frac{\partial \phi_i}{\partial x_c} = \sum_r \xi_{r,c}\, D_r,
 * \f]
 * (\f$ \xi_{r,c} = \partial \xi_r / \partial x_c \f$, \f$ C \f$ from
 * @ref specfem::medium_physics::constitutive_tensor), written as
 * \f[
 *   M(a, b, r, s; q) = \sum_{c,d} \xi_{r,c}(q)\, C_{a c b d}(q)\,
 *   \xi_{s,d}(q)\, w(q) J(q), \qquad
 *   K_e(a, i; b, j) = \sum_{r, s, q} D(r, q, i)\, M(a, b, r, s; q)\,
 *   D(s, q, j).
 * \f]
 * \f$ D(r, q, i) = \partial \phi_i / \partial \xi_r \f$ at GLL point
 * \f$ q \f$ is collocated: \f$ h(q_r, i_r) \f$ times Kronecker deltas
 * \f$ \delta(q_t, i_t) \f$ along the two other directions. It is passed as
 * a delta-structured operand, so TensorOperations eliminates the deltas at
 * compile time and each entry costs at most one length-NGLL sum (the
 * sum-factored form) -- the expression stays the one above.
 *
 * The graph, one team per (element, row component \f$ a \f$, column
 * component \f$ b \f$), three levels:
 * 1. **Stage** the Lagrange derivative matrix \f$ h(u, f) \f$ =
 *    `hprime(point, function)`.
 * 2. **Einsum** \f$ M \f$ over \f$ (c, d) \f$ from three single-entry
 *    functional leaves (\f$ \xi \f$, \f$ C \f$, \f$ w J \f$), read at
 *    the global coordinate and never staged.
 * 3. **Einsum** \f$ K_e \f$ over \f$ (r, s, z, y, x) \f$. The output is a
 *    rank-9 alias of `k_e` with axes (\f$ e, a, k, j, i, b, n, m, l \f$) =
 *    (element, row component, row node \f$ z, y, x \f$, column component,
 *    column node \f$ z, y, x \f$) -- exactly the
 *    @ref specfem::linear_system::local_dof_index ordering, so there is no
 *    reshape, no identity input and no workspace.
 *
 * Cost per element is \f$ O(N_{GLL}^5) \f$ arithmetic against an
 * \f$ O(N_{GLL}^6) \f$ block write, so the kernel is write-bound. Sign
 * convention as the probe kernel: \f$ K u \f$ is the internal force.
 * Medium-generic: \f$ a, b \f$ are single labels of extent `ncomp`; the
 * only medium-specific code is the constitutive accessor.
 *
 * Team scratch is level 0 on GPU and level 1 on host backends (their level-0
 * cap of 32 KB is below the whole-tile output block).
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
class StiffnessDirectKernel {
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
   * @brief Bind the kernel to an assembly.
   *
   * Throws `std::runtime_error` if the mesh grid does not match `NGLL`. No
   * workspace is allocated: the graph lives in team scratch.
   *
   * @param assembly Assembled mesh, jacobian matrix, and material properties;
   *        borrowed, must outlive the kernel
   */
  explicit StiffnessDirectKernel(const AssemblyType &assembly);

  /**
   * @brief Fill the leading `batch.size()` blocks of `k_e`.
   *
   * Does not fence: a caller that consumes `k_e` on device must fence first
   * (a host mirror copy synchronizes by itself).
   *
   * @param batch Contiguous element sub-range [begin, end) in compute-domain
   *        indices; the block for element `ispec` lands at slot
   *        `ispec - batch.begin_index()`
   * @param k_e Preallocated device buffer of shape `(>= batch size, ndof,
   *            ndof)`; same contract as
   *            @ref specfem::linear_system::compute_element_stiffness
   */
  void operator()(const specfem::datatype::ElementIndexRange &batch,
                  const StiffnessViewType &k_e) const;

private:
  const AssemblyType &assembly_; ///< Borrowed assembly (not owned)
};

/**
 * @brief One-shot wrapper over @ref StiffnessDirectKernel:
 * construct-and-call, then fence, matching
 * @ref specfem::linear_system::compute_element_stiffness.
 */
template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
void compute_element_stiffness_direct(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e);

} // namespace specfem::linear_system_impl
