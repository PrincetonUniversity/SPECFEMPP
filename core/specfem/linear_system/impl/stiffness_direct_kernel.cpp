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
#include <TensorOperations/StridedAlias.hpp>
#include <TensorOperations/Tiling.hpp>
#include <sstream>
#include <stdexcept>

namespace specfem::linear_system_impl {

/**
 * @brief Leaf of the weighted reference-frame constitutive tensor
 * \f$ M_{rs}(a, b; q) = w(q) J(q) \sum_{c,d} \xi_{r,c} C_{acbd}
 * \xi_{s,d} \f$ at the global coordinate (element slot `e`, components `a`,
 * `b`, reference directions `r`, `s`, quadrature point `z, y, x`).
 *
 * \f$ \xi_{r,c} = \partial \xi_r / \partial x_c \f$ is
 * `jacobian_matrix.tensor()(c, r)` (row = spatial, column = reference);
 * \f$ C \f$ is @ref specfem::medium_physics::constitutive_tensor.
 */
template <typename Tags, typename JacobianMatrixType, typename PropertiesType,
          typename WeightsViewType>
struct WeightedReferenceConstitutiveLeaf {
  using PointTags =
      specfem::tags::Tags<Tags::dimension_tag, Tags::medium_tag,
                          Tags::property_tag, Tags::attenuation_tag, false>;
  using PointIndexType = specfem::point::index<Tags::dimension_tag, false>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<Tags::dimension_tag, true, false>;
  using PointPropertyType = specfem::point::properties<PointTags>;

  JacobianMatrixType jacobian_matrix;
  PropertiesType properties;
  WeightsViewType weights;
  int batch_begin_ispec;

  KOKKOS_FUNCTION type_real operator()(const int e, const int a, const int b,
                                       const int r, const int s, const int z,
                                       const int y, const int x) const {
    constexpr int ndim = 3;
    const PointIndexType index(batch_begin_ispec + e, z, y, x);

    PointJacobianMatrixType point_jacobian_matrix;
    specfem::assembly::load_on_device(index, jacobian_matrix,
                                      point_jacobian_matrix);
    PointPropertyType point_property;
    specfem::assembly::load_on_device(index, properties, point_property);

    const auto &xi = point_jacobian_matrix.tensor();
    type_real sum = 0;
    for (int c = 0; c < ndim; ++c) {
      for (int d = 0; d < ndim; ++d) {
        sum += xi(c, r) *
               specfem::medium_physics::constitutive_tensor<PointTags>(
                   point_property, a, c, b, d) *
               xi(d, s);
      }
    }
    return weights(x) * weights(y) * weights(z) *
           point_jacobian_matrix.jacobian() * sum;
  }
};

/**
 * @brief The sum-factored element stiffness, one (r, s) term at a time.
 *
 * Output coordinate (element slot `e`, row component `a`, row point
 * `k, j, i` = z, y, x, column component `b`, column point `n, m, l`),
 * reduction coordinate `(r, s)`. `h(q, f)` is the Lagrange derivative matrix
 * (point, function), fully free; `M(z, y, x)` is the constitutive tensor
 * bound on `e, a, b, r, s`. The body is the closed form term for term:
 * the directions outside \f$ \{r, s\} \f$ must coincide; for \f$ r = s \f$
 * the quadrature index runs along direction \f$ r \f$; for \f$ r \ne s \f$
 * it is pinned to the column's \f$ r \f$ coordinate.
 */
template <int NGLL> struct SumFactoredStiffness {
  template <typename H, typename M>
  KOKKOS_FUNCTION void
  operator()(int /* e */, int /* a */, const int k, const int j, const int i,
             int /* b */, const int n, const int m, const int l, const int r,
             const int s, const H &h, const M &M_rs, type_real &acc) const {
    const int row[3] = { i, j, k }; // x, y, z
    const int col[3] = { l, m, n };
    for (int t = 0; t < 3; ++t) {
      if (t != r && t != s && row[t] != col[t]) {
        return;
      }
    }
    int p[3] = { row[0], row[1], row[2] }; // quadrature point (x, y, z)
    if (r == s) {
      for (int q = 0; q < NGLL; ++q) {
        p[r] = q;
        acc += h(q, row[r]) * M_rs(p[2], p[1], p[0]) * h(q, col[r]);
      }
    } else {
      p[r] = col[r];
      acc += h(col[r], row[r]) * M_rs(p[2], p[1], p[0]) * h(row[s], col[s]);
    }
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

  // The graph output IS k_e: a rank-9 alias over the same memory with axes
  // (e, a, k, j, i, b, n, m, l). Row = local_dof_index(a, k, j, i) and
  // column = local_dof_index(b, n, m, l), LayoutRight, so the column point's
  // x index is fastest. Kokkos::View stops at rank 8; TensorOperations only
  // needs a TensorLike, which StridedAlias is.
  namespace tenops = TensorOperations;
  const auto k_alias =
      tenops::make_strided_alias<ExecSpace, ncomp, NGLL, NGLL, NGLL, ncomp,
                                 NGLL, NGLL, NGLL>(k_e.data(), nbatch);

  const specfem::linear_system_impl::WeightedReferenceConstitutiveLeaf<
      Tags, JacobianMatrixType, PropertiesType, WeightsViewType>
      constitutive{ assembly_.jacobian_matrix, assembly_.properties,
                    assembly_.mesh.weights, batch_begin_ispec };

  // Labels: e element slot, a/b row/column component (gridded: one team per
  // (e, a, b)); k, j, i the row point and n, m, l the column point (z, y, x);
  // r, s reference directions (the reduction); z, y, x the quadrature point
  // of M; q, f the point and function axes of h.
  using TileMap = tenops::LabelTiles<
      tenops::LabelTile<'e', 1>, tenops::LabelTile<'a', 1>,
      tenops::LabelTile<'b', 1>, tenops::LabelWhole<'k', NGLL>,
      tenops::LabelWhole<'j', NGLL>, tenops::LabelWhole<'i', NGLL>,
      tenops::LabelWhole<'n', NGLL>, tenops::LabelWhole<'m', NGLL>,
      tenops::LabelWhole<'l', NGLL>, tenops::LabelWhole<'r', ndim>,
      tenops::LabelWhole<'s', ndim>, tenops::LabelWhole<'z', NGLL>,
      tenops::LabelWhole<'y', NGLL>, tenops::LabelWhole<'x', NGLL>,
      tenops::LabelWhole<'q', NGLL>, tenops::LabelWhole<'f', NGLL>>;

  auto g0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});

  // h(q, f): the Lagrange derivative matrix, hprime(point, function).
  auto [g1, h] = g0.add(tenops::make_stage_node(tenops::make_input_node(
      tenops::make_handle<'q', 'f'>(assembly_.mesh.hprime))));

  // M(e, a, b, r, s, z, y, x) = w J sum_{c,d} xi_{r,c} C_{acbd} xi_{s,d}.
  auto [g2, M] = g1.add(tenops::make_stage_node(
      tenops::make_functional_input_node<'e', 'a', 'b', 'r', 's', 'z', 'y',
                                         'x'>(
          Kokkos::Array<int, 8>{ nbatch, ncomp, ncomp, ndim, ndim, NGLL, NGLL,
                                 NGLL },
          constitutive)));

  // K(e, a, k, j, i, b, n, m, l) = sum_{r,s} (the closed form).
  auto [g3, K] = g2.add(
      tenops::make_reduce_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(
          tenops::over<'r', 's'>{}, h, M,
          specfem::linear_system_impl::SumFactoredStiffness<NGLL>{}));

  // Instantiating the plan runs LevelGraph's structural guards.
  using Plan = tenops::LevelPlan<std::decay_t<decltype(g3.levels)>>;
  static_assert(Plan::num_levels == 3, "two stage levels and one reduce");

  // Host backends cap level-0 team scratch at 32 KB, below the whole-tile
  // output block; level 1 allows tens of MB. On GPU level 0 is on-chip.
  constexpr bool on_gpu =
      !Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible;
  g3.outputs(K)
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
