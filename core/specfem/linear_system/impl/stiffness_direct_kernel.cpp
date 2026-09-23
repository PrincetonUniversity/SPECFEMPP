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
#include <TensorOperations/Einsum.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/StridedAlias.hpp>
#include <TensorOperations/Tiling.hpp>
#include <sstream>
#include <stdexcept>

namespace specfem::linear_system_impl {

/**
 * @brief Leaf \f$ \xi_{r,c} = \partial \xi_r / \partial x_c \f$ at the
 * global coordinate (element slot `e`, quadrature point `z, y, x`, reference
 * direction `r`, spatial direction `c`): `jacobian_matrix.tensor()(c, r)`
 * (row = spatial, column = reference).
 */
template <typename Tags, typename JacobianMatrixType>
struct ReferenceGradientLeaf {
  using PointIndexType = specfem::point::index<Tags::dimension_tag, false>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<Tags::dimension_tag, true, false>;

  JacobianMatrixType jacobian_matrix;
  int batch_begin_ispec;

  KOKKOS_FUNCTION type_real operator()(const int e, const int z, const int y,
                                       const int x, const int r,
                                       const int c) const {
    const PointIndexType index(batch_begin_ispec + e, z, y, x);
    PointJacobianMatrixType point_jacobian_matrix;
    specfem::assembly::load_on_device(index, jacobian_matrix,
                                      point_jacobian_matrix);
    return point_jacobian_matrix.tensor()(c, r);
  }
};

/**
 * @brief Leaf \f$ C_{acbd} \f$ at the global coordinate (element slot `e`,
 * quadrature point `z, y, x`), from
 * @ref specfem::medium_physics::constitutive_tensor.
 */
template <typename Tags, typename PropertiesType> struct ConstitutiveLeaf {
  using PointTags =
      specfem::tags::Tags<Tags::dimension_tag, Tags::medium_tag,
                          Tags::property_tag, Tags::attenuation_tag, false>;
  using PointIndexType = specfem::point::index<Tags::dimension_tag, false>;
  using PointPropertyType = specfem::point::properties<PointTags>;

  PropertiesType properties;
  int batch_begin_ispec;

  KOKKOS_FUNCTION type_real operator()(const int e, const int z, const int y,
                                       const int x, const int a, const int c,
                                       const int b, const int d) const {
    const PointIndexType index(batch_begin_ispec + e, z, y, x);
    PointPropertyType point_property;
    specfem::assembly::load_on_device(index, properties, point_property);
    return specfem::medium_physics::constitutive_tensor<PointTags>(
        point_property, a, c, b, d);
  }
};

/**
 * @brief Leaf \f$ w(x) w(y) w(z) J \f$ (quadrature weight times Jacobian
 * determinant) at the global coordinate (element slot `e`, quadrature point
 * `z, y, x`).
 */
template <typename Tags, typename JacobianMatrixType, typename WeightsViewType>
struct QuadratureWeightLeaf {
  using PointIndexType = specfem::point::index<Tags::dimension_tag, false>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<Tags::dimension_tag, true, false>;

  JacobianMatrixType jacobian_matrix;
  WeightsViewType weights;
  int batch_begin_ispec;

  KOKKOS_FUNCTION type_real operator()(const int e, const int z, const int y,
                                       const int x) const {
    const PointIndexType index(batch_begin_ispec + e, z, y, x);
    PointJacobianMatrixType point_jacobian_matrix;
    specfem::assembly::load_on_device(index, jacobian_matrix,
                                      point_jacobian_matrix);
    return weights(x) * weights(y) * weights(z) *
           point_jacobian_matrix.jacobian();
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

  // Labels: e element slot, a/b row/column component (gridded: one team per
  // (e, a, b)); k, j, i the row node and n, m, l the column node (z, y, x);
  // z, y, x the quadrature point; r, s reference directions; c, d spatial
  // directions; u, f the point and function axes of h.
  using TileMap = tenops::LabelTiles<
      tenops::LabelTile<'e', 1>, tenops::LabelTile<'a', 1>,
      tenops::LabelTile<'b', 1>, tenops::LabelWhole<'k', NGLL>,
      tenops::LabelWhole<'j', NGLL>, tenops::LabelWhole<'i', NGLL>,
      tenops::LabelWhole<'n', NGLL>, tenops::LabelWhole<'m', NGLL>,
      tenops::LabelWhole<'l', NGLL>, tenops::LabelWhole<'z', NGLL>,
      tenops::LabelWhole<'y', NGLL>, tenops::LabelWhole<'x', NGLL>,
      tenops::LabelWhole<'r', ndim>, tenops::LabelWhole<'s', ndim>,
      tenops::LabelWhole<'c', ndim>, tenops::LabelWhole<'d', ndim>,
      tenops::LabelWhole<'u', NGLL>, tenops::LabelWhole<'f', NGLL>>;

  const Kokkos::Array<int, 6> xi_extents{
    nbatch, NGLL, NGLL, NGLL, ndim, ndim
  };
  const specfem::linear_system_impl::ReferenceGradientLeaf<Tags,
                                                           JacobianMatrixType>
      xi_leaf{ assembly_.jacobian_matrix, batch_begin_ispec };
  const specfem::linear_system_impl::ConstitutiveLeaf<Tags, PropertiesType>
      c_leaf{ assembly_.properties, batch_begin_ispec };
  const specfem::linear_system_impl::QuadratureWeightLeaf<
      Tags, JacobianMatrixType, WeightsViewType>
      w_leaf{ assembly_.jacobian_matrix, assembly_.mesh.weights,
              batch_begin_ispec };

  const auto xi =
      tenops::make_functional_input_node<'e', 'z', 'y', 'x', 'r', 'c'>(
          xi_extents, xi_leaf);
  const auto C = tenops::make_functional_input_node<'e', 'z', 'y', 'x', 'a',
                                                    'c', 'b', 'd'>(
      Kokkos::Array<int, 8>{ nbatch, NGLL, NGLL, NGLL, ncomp, ndim, ncomp,
                             ndim },
      c_leaf);
  const auto wJ = tenops::make_functional_input_node<'e', 'z', 'y', 'x'>(
      Kokkos::Array<int, 4>{ nbatch, NGLL, NGLL, NGLL }, w_leaf);

  auto g0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});

  // h(u, f): the Lagrange derivative matrix, hprime(point, function).
  auto [g1, h] = g0.add(tenops::make_stage_node(tenops::make_input_node(
      tenops::make_handle<'u', 'f'>(assembly_.mesh.hprime))));

  // The element's reference gradients and weights are read NGLL^3 * 9 times
  // each by M, so they are staged once; C is read once per term and stays a
  // functional leaf. One stage level per tile shape.
  auto [g2, xi_rc] = g1.add(tenops::make_stage_node(xi));
  auto [g3, w] = g2.add(tenops::make_stage_node(wJ));

  // M(e, a, b, r, s, z, y, x) = sum_{c,d} xi_{r,c} C_{acbd} xi_{s,d} w J
  auto [g4, M] =
      g3.add(tenops::make_einsum_node<'e', 'a', 'b', 'r', 's', 'z', 'y', 'x'>(
          xi_rc, C, xi_rc.template as<'e', 'z', 'y', 'x', 's', 'd'>(), w));

  // D(r, z, y, x, k, j, i) = d phi_{kji} / d xi_r at (z, y, x): h along r,
  // Kronecker deltas along the other two directions.
  const auto D = tenops::make_delta_operand<'r', 'z', 'y', 'x', 'k', 'j', 'i'>(
      tenops::select<'r'>(
          tenops::kase(h.template as<'x', 'i'>(), tenops::delta<'y', 'j'>{},
                       tenops::delta<'z', 'k'>{}),
          tenops::kase(h.template as<'y', 'j'>(), tenops::delta<'x', 'i'>{},
                       tenops::delta<'z', 'k'>{}),
          tenops::kase(h.template as<'z', 'k'>(), tenops::delta<'x', 'i'>{},
                       tenops::delta<'y', 'j'>{})));

  // K(e, a, k, j, i, b, n, m, l)
  //   = sum_{r,s,z,y,x} D(r, z, y, x, k, j, i) M(e, a, b, r, s, z, y, x)
  //                     D(s, z, y, x, n, m, l)
  auto [g5, K] = g4.add(
      tenops::make_einsum_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(
          D, M, D.template as<'s', 'z', 'y', 'x', 'n', 'm', 'l'>()));

  // Instantiating the plan runs LevelGraph's structural guards.
  using Plan = tenops::LevelPlan<std::decay_t<decltype(g5.levels)>>;
  static_assert(Plan::num_levels == 5, "stage h, xi, w J; einsum M; einsum K");

  // Host backends cap level-0 team scratch at 32 KB, below the whole-tile
  // output block; level 1 allows tens of MB. On GPU level 0 is on-chip.
  constexpr bool on_gpu =
      !Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible;
  g5.outputs(K)
      .scratch_level(on_gpu ? 0 : 1)
      .execute(tenops::TeamPolicyTag<ExecSpace>{}, k_alias);
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
