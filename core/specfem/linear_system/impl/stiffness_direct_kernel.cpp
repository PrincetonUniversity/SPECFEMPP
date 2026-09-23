// One of the two translation units that include TensorOperations headers
// (the other is stiffness_tensor_graph_kernel.cpp). Without
// SPECFEM_ENABLE_TENSOROPS the entry points are throwing stubs (bottom of the
// file), so callers dispatch without preprocessor branches.
#include "specfem/linear_system/impl/stiffness_direct_kernel.hpp"

#include "specfem/linear_system/element_stiffness.hpp"

#ifdef SPECFEM_ENABLE_TENSOROPS

#include "specfem/assembly/assembly.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/Tiling.hpp>
#include <sstream>
#include <stdexcept>

namespace specfem::linear_system_impl {

/// Point-level types shared by the leaves (scalar, no SIMD).
template <typename Tags> struct DirectKernelPointTypes {
  using PointTags =
      specfem::tags::Tags<Tags::dimension_tag, Tags::medium_tag,
                          Tags::property_tag, Tags::attenuation_tag, false>;
  using PointIndexType = specfem::point::index<Tags::dimension_tag, false>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<Tags::dimension_tag, true, false>;
  using PointPropertyType = specfem::point::properties<PointTags>;
  constexpr static int ncomp =
      specfem::element::attributes<Tags::dimension_tag,
                                   Tags::medium_tag>::components;
};

/**
 * @brief Leaf \f$ \xi_{r,c} = \partial \xi_r / \partial x_c \f$ at the global
 * coordinate (row block `E`, spatial `c`, reference `r`, point `z, y, x`).
 */
template <typename Tags, typename JacobianMatrixType>
struct InverseJacobianLeaf {
  using Types = DirectKernelPointTypes<Tags>;
  JacobianMatrixType jacobian_matrix;
  int batch_begin_ispec;

  KOKKOS_FUNCTION type_real operator()(const int E, const int c, const int r,
                                       const int z, const int y,
                                       const int x) const {
    const typename Types::PointIndexType index(
        batch_begin_ispec + E / Types::ncomp, z, y, x);
    typename Types::PointJacobianMatrixType point_jacobian_matrix;
    specfem::assembly::load_on_device(index, jacobian_matrix,
                                      point_jacobian_matrix);
    // tensor()(c, r): row = spatial coordinate, column = reference
    // coordinate (see specfem::point::jacobian_matrix).
    return point_jacobian_matrix.tensor()(c, r);
  }
};

/**
 * @brief Leaf \f$ C_{a c b d} \f$ at the global coordinate, with the row
 * component \f$ a = E \bmod n_{comp} \f$ and the column component `b`.
 */
template <typename Tags, typename PropertiesType> struct ConstitutiveLeaf {
  using Types = DirectKernelPointTypes<Tags>;
  PropertiesType properties;
  int batch_begin_ispec;

  KOKKOS_FUNCTION type_real operator()(const int E, const int b, const int c,
                                       const int d, const int z, const int y,
                                       const int x) const {
    const typename Types::PointIndexType index(
        batch_begin_ispec + E / Types::ncomp, z, y, x);
    typename Types::PointPropertyType point_property;
    specfem::assembly::load_on_device(index, properties, point_property);
    return specfem::medium_physics::constitutive_tensor<
        typename Types::PointTags>(point_property, E % Types::ncomp, c, b, d);
  }
};

/// Leaf \f$ w(q) J(q) \f$ at the global coordinate.
template <typename Tags, typename JacobianMatrixType, typename WeightsViewType>
struct WeightedJacobianLeaf {
  using Types = DirectKernelPointTypes<Tags>;
  JacobianMatrixType jacobian_matrix;
  WeightsViewType weights;
  int batch_begin_ispec;

  KOKKOS_FUNCTION type_real operator()(const int E, const int z, const int y,
                                       const int x) const {
    const typename Types::PointIndexType index(
        batch_begin_ispec + E / Types::ncomp, z, y, x);
    typename Types::PointJacobianMatrixType point_jacobian_matrix;
    specfem::assembly::load_on_device(index, jacobian_matrix,
                                      point_jacobian_matrix);
    return weights(x) * weights(y) * weights(z) *
           point_jacobian_matrix.jacobian();
  }
};

/**
 * @brief Level-1 reduce over \f$ (c, d) \f$:
 * \f$ M_{rs} = w J \, \xi_{r,c} \, C_{a c b d} \, \xi_{s,d} \f$. The
 * coordinates (7 output, 2 reduction) are unused: the product is pointwise.
 */
struct ReferenceConstitutive {
  KOKKOS_FUNCTION type_real operator()(int, int, int, int, int, int, int, int,
                                       int, const type_real wJ,
                                       const type_real xi_rc, const type_real C,
                                       const type_real xi_sd) const {
    return wJ * xi_rc * C * xi_sd;
  }
};

/**
 * @brief Level-2 reduce over \f$ (q, s) \f$: the nine sum-factored terms for
 * the current column direction `s`, one per row direction `r`.
 *
 * Output coordinate `(E, k, j, i, b, K, J, I)` = (row block, \f$ i_z, i_y,
 * i_x \f$, column component, \f$ j_z, j_y, j_x \f$); reduction coordinate
 * `(q, s)` in first-appearance order over the operand list. Operands: the
 * nine relabels of \f$ h \f$ (point first, function second), then
 * \f$ M_{x s}, M_{y s}, M_{z s} \f$ read with `q` in the `r` slot.
 */
struct SumFactoredStiffness {
  KOKKOS_FUNCTION type_real
  operator()(int, const int k, const int j, const int i, int, const int K,
             const int J, const int I, const int q, const int s,
             const type_real hqi, const type_real hqj, const type_real hqk,
             const type_real hqI, const type_real hqJ, const type_real hqK,
             const type_real hiI, const type_real hjJ, const type_real hkK,
             const type_real mx, const type_real my, const type_real mz) const {
    const bool dk = (k == K);
    const bool dj = (j == J);
    const bool di = (i == I);
    const type_real zero = static_cast<type_real>(0.0);

    // r = x: row factor h(q, i_x); M_x(k, j, q)
    type_real sx = zero;
    if (s == 0) {
      sx = (dj && dk) ? hqI : zero;
    } else if (s == 1) {
      sx = (dk && q == I) ? hjJ : zero;
    } else {
      sx = (dj && q == I) ? hkK : zero;
    }
    // r = y: row factor h(q, i_y); M_y(k, q, i)
    type_real sy = zero;
    if (s == 0) {
      sy = (dk && q == J) ? hiI : zero;
    } else if (s == 1) {
      sy = (di && dk) ? hqJ : zero;
    } else {
      sy = (di && q == J) ? hkK : zero;
    }
    // r = z: row factor h(q, i_z); M_z(q, j, i)
    type_real sz = zero;
    if (s == 0) {
      sz = (dj && q == K) ? hiI : zero;
    } else if (s == 1) {
      sz = (di && q == K) ? hjJ : zero;
    } else {
      sz = (di && dj) ? hqK : zero;
    }
    return hqi * mx * sx + hqj * my * sy + hqk * mz * sz;
  }
};

} // namespace specfem::linear_system_impl

template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
specfem::linear_system_impl::StiffnessDirectKernel<
    NGLL, Tags>::StiffnessDirectKernel(const AssemblyType &assembly)
    : assembly_(assembly) {
  if (assembly.mesh.element_grid != NGLL) {
    throw std::runtime_error(
        "specfem::linear_system_impl::StiffnessDirectKernel: the number of "
        "GLL points in the mesh elements must match the template parameter "
        "NGLL.");
  }
}

template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
void specfem::linear_system_impl::StiffnessDirectKernel<NGLL, Tags>::operator()(
    const specfem::datatype::ElementIndexRange &batch,
    const StiffnessViewType &k_e) const {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using JacobianMatrixType = std::decay_t<decltype(assembly_.jacobian_matrix)>;
  using PropertiesType = std::decay_t<decltype(assembly_.properties)>;
  using WeightsViewType = std::decay_t<decltype(assembly_.mesh.weights)>;

  if (batch.empty()) {
    return;
  }

  const int nbatch = batch.size();
  const int batch_begin_ispec = batch.begin_index();

  if (static_cast<int>(k_e.extent(0)) < nbatch ||
      static_cast<int>(k_e.extent(1)) != ndof ||
      static_cast<int>(k_e.extent(2)) != ndof) {
    throw std::runtime_error(
        "specfem::linear_system_impl::StiffnessDirectKernel: the element "
        "stiffness buffer must have extents (>= batch size, ndof, ndof) "
        "with ndof = ncomp * NGLL^3.");
  }

  constexpr int ndim = 3;
  const int nrows = nbatch * ncomp; // row blocks E = e * ncomp + a

  // The graph output IS k_e: a rank-8 alias over the same memory with axes
  // (E, iz, iy, ix, b, jz, jy, jx). Row = local_dof_index(a, iz, iy, ix) and
  // column = local_dof_index(b, jz, jy, jx), so with ndof = ncomp * NGLL^3
  // the block row offset E * NGLL^3 * ndof is exactly (e * ndof + a *
  // NGLL^3) * ndof. LayoutRight keeps the column index fastest.
  using OutputAliasType =
      Kokkos::View<type_real *[NGLL][NGLL][NGLL][ncomp][NGLL][NGLL][NGLL],
                   Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const OutputAliasType k_alias(k_e.data(), nrows);

  const specfem::linear_system_impl::InverseJacobianLeaf<Tags,
                                                         JacobianMatrixType>
      xi_leaf{ assembly_.jacobian_matrix, batch_begin_ispec };
  const specfem::linear_system_impl::ConstitutiveLeaf<Tags, PropertiesType>
      c_leaf{ assembly_.properties, batch_begin_ispec };
  const specfem::linear_system_impl::WeightedJacobianLeaf<
      Tags, JacobianMatrixType, WeightsViewType>
      wj_leaf{ assembly_.jacobian_matrix, assembly_.mesh.weights,
               batch_begin_ispec };

  // Labels: E row block, b column component (both gridded, one team each);
  // k/j/i the row point's z/y/x GLL index, K/J/I the column point's; q the
  // summed quadrature index, s the summed column direction; r/c/d/z/y/x the
  // internal axes of M and its leaves. Every GLL label stays whole: the
  // output tile is the full (NGLL^6) block of one (E, b) pair.
  namespace tenops = TensorOperations;
  using TileMap = tenops::LabelTiles<
      tenops::LabelTile<'E', 1>, tenops::LabelTile<'b', 1>,
      tenops::LabelWhole<'k', NGLL>, tenops::LabelWhole<'j', NGLL>,
      tenops::LabelWhole<'i', NGLL>, tenops::LabelWhole<'K', NGLL>,
      tenops::LabelWhole<'J', NGLL>, tenops::LabelWhole<'I', NGLL>,
      tenops::LabelWhole<'q', NGLL>, tenops::LabelWhole<'s', ndim>,
      tenops::LabelWhole<'r', ndim>, tenops::LabelWhole<'c', ndim>,
      tenops::LabelWhole<'d', ndim>, tenops::LabelWhole<'z', NGLL>,
      tenops::LabelWhole<'y', NGLL>, tenops::LabelWhole<'x', NGLL>>;

  auto g0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});

  // Stage levels: one per distinct tile shape.
  auto [g1, h] = g0.add(tenops::make_stage_node(tenops::make_input_node(
      tenops::make_handle<'q', 'i'>(assembly_.mesh.hprime))));
  auto [g2, xi] = g1.add(tenops::make_stage_node(
      tenops::make_functional_input_node<'E', 'c', 'r', 'z', 'y', 'x'>(
          Kokkos::Array<int, 6>{ nrows, ndim, ndim, NGLL, NGLL, NGLL },
          xi_leaf)));
  auto [g3, C] = g2.add(tenops::make_stage_node(
      tenops::make_functional_input_node<'E', 'b', 'c', 'd', 'z', 'y', 'x'>(
          Kokkos::Array<int, 7>{ nrows, ncomp, ndim, ndim, NGLL, NGLL, NGLL },
          c_leaf)));
  auto [g4, wJ] = g3.add(tenops::make_stage_node(
      tenops::make_functional_input_node<'E', 'z', 'y', 'x'>(
          Kokkos::Array<int, 4>{ nrows, NGLL, NGLL, NGLL }, wj_leaf)));

  // M(E, b, r, s, z, y, x) = sum_{c,d} wJ xi(c, r) C(c, d) xi(d, s):
  // reduction labels (c, d) in first-appearance order.
  auto [g5, M] =
      g4.add(tenops::make_reduce_node<'E', 'b', 'r', 's', 'z', 'y', 'x'>(
          wJ, xi.template as<'E', 'c', 'r', 'z', 'y', 'x'>(), C,
          xi.template as<'E', 'd', 's', 'z', 'y', 'x'>(),
          specfem::linear_system_impl::ReferenceConstitutive{}));

  // K(E, k, j, i, b, K, J, I) = sum_{q,s} ...: the nine relabels of h (point,
  // function), then M_x, M_y, M_z read with q in the r slot. Reduction
  // labels (q, s) in first-appearance order.
  auto [g6, K] =
      g5.add(tenops::make_reduce_node<'E', 'k', 'j', 'i', 'b', 'K', 'J', 'I'>(
          h.template as<'q', 'i'>(), h.template as<'q', 'j'>(),
          h.template as<'q', 'k'>(), h.template as<'q', 'I'>(),
          h.template as<'q', 'J'>(), h.template as<'q', 'K'>(),
          h.template as<'i', 'I'>(), h.template as<'j', 'J'>(),
          h.template as<'k', 'K'>(),
          M.template as<'E', 'b', tenops::fixed<0>, 's', 'k', 'j', 'q'>(),
          M.template as<'E', 'b', tenops::fixed<1>, 's', 'k', 'q', 'i'>(),
          M.template as<'E', 'b', tenops::fixed<2>, 's', 'q', 'j', 'i'>(),
          specfem::linear_system_impl::SumFactoredStiffness{}));

  // Instantiating the plan runs LevelGraph's structural guards.
  using Plan = tenops::LevelPlan<std::decay_t<decltype(g6.levels)>>;
  static_assert(Plan::num_levels == 6, "four stage levels and two reduces");

  // Host backends cap level-0 team scratch at 32 KB, below the whole-tile
  // output block; level 1 allows tens of MB. On GPU level 0 is on-chip.
  constexpr bool on_gpu =
      !Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible;
  g6.outputs(K)
      .scratch_level(on_gpu ? 0 : 1)
      .execute(tenops::TeamPolicyTag2<ExecSpace>{}, k_alias);
  // No fence: see the call operator's contract in the header.
}

#else // !SPECFEM_ENABLE_TENSOROPS

#include <stdexcept>

namespace specfem::linear_system_impl {
/// Single throw message for every stub of the OFF build.
inline constexpr const char *direct_kernel_unavailable_message =
    "specfem::linear_system::compute_element_stiffness: the direct kernel "
    "requires SPECFEM++ built with SPECFEM_ENABLE_TENSOROPS=ON (and "
    "SPECFEM_TENSOROPS_ROOT pointing at a TensorOperations checkout).";
} // namespace specfem::linear_system_impl

template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
specfem::linear_system_impl::StiffnessDirectKernel<
    NGLL, Tags>::StiffnessDirectKernel(const AssemblyType &assembly)
    : assembly_(assembly) {
  throw std::runtime_error(
      specfem::linear_system_impl::direct_kernel_unavailable_message);
}

template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
void specfem::linear_system_impl::StiffnessDirectKernel<NGLL, Tags>::operator()(
    const specfem::datatype::ElementIndexRange & /* batch */,
    const StiffnessViewType & /* k_e */) const {
  throw std::runtime_error(
      specfem::linear_system_impl::direct_kernel_unavailable_message);
}

#endif // SPECFEM_ENABLE_TENSOROPS

// The one-shot wrapper and the explicit instantiations are shared by both
// builds: without TensorOperations the constructor above throws.
template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
void specfem::linear_system_impl::compute_element_stiffness_direct(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e) {
  const specfem::linear_system_impl::StiffnessDirectKernel<NGLL, Tags> kernel(
      assembly);
  kernel(batch, k_e);
  Kokkos::fence();
}

// Explicit instantiations: 3D elastic isotropic, NGLL = 5 (mirrors
// element_stiffness.cpp).
template class specfem::linear_system_impl::StiffnessDirectKernel<
    5, specfem::linear_system_impl::elastic_isotropic_tags>;

template void specfem::linear_system_impl::compute_element_stiffness_direct<
    5, specfem::linear_system_impl::elastic_isotropic_tags>(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const specfem::datatype::ElementIndexRange &,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &);
