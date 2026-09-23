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
 * every entry in closed form through one TensorOperations level graph whose
 * last node is a reduction over quadrature.
 *
 * The block is the bilinear form \f$ K_e = \sum_q D_q^T (w J C)_q D_q \f$.
 * GLL derivatives are collocated, so the reference derivative of basis
 * \f$ i = (i_x, i_y, i_z) \f$ along direction \f$ r \f$ at quadrature point
 * \f$ q \f$ is \f$ h(q_r, i_r) \f$ times \f$ [q_t = i_t] \f$ for the two other
 * directions, and after sum factoring each entry is at most one length-NGLL
 * sum. With the weighted reference-frame constitutive tensor
 * \f[
 *   M_{r s}(a, b; q) = w(q) J(q) \sum_{c,d} \xi_{r,c}(q) C_{a c b d}(q)
 *   \xi_{s,d}(q)
 * \f]
 * (\f$ \xi_{r,c} = \partial \xi_r / \partial x_c \f$, \f$ C \f$ from
 * @ref specfem::medium_physics::constitutive_tensor), the entry at row
 * \f$ (a, i) \f$ and column \f$ (b, j) \f$ is
 * \f[
 *   K_e = \sum_{r,s} \sum_q h(q, i_r) \, M_{r s}(a, b; q \text{ in slot } r,
 *   i_t \text{ elsewhere}) \, S_{r s}, \quad
 *   S_{r s} = \begin{cases}
 *     h(q, j_r) \prod_{t \ne r} [i_t = j_t] & s = r \\
 *     [q = j_r] \, h(i_s, j_s) \prod_{t \ne r, s} [i_t = j_t] & s \ne r
 *   \end{cases}
 * \f]
 *
 * The graph, one team per (element, row component \f$ a \f$, column
 * component \f$ b \f$), three levels:
 * 1. **Stage** the Lagrange derivative matrix \f$ h(q, f) \f$.
 * 2. **Stage** \f$ M_{rs}(a, b; z, y, x) \f$ from one functional leaf whose
 *    functor is the formula above (Jacobian, material, weights at the global
 *    coordinate; the \f$ (c, d) \f$ sum as two loops).
 * 3. **Reduce** over \f$ (r, s) \f$ (`TensorOperations::make_reduce_node`, a
 *    parallel_reduce-shaped node): the functor receives `h` and `M` as
 *    accessors (`M` bound on \f$ e, a, b, r, s \f$) and adds the
 *    \f$ (r, s) \f$ term of the closed form, loading only what it uses. The
 *    output is a rank-9 alias of `k_e` with axes (\f$ e, a, k, j, i, b, n,
 *    m, l \f$) = (element, row component, row point \f$ z, y, x \f$, column
 *    component, column point \f$ z, y, x \f$) -- exactly the
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
