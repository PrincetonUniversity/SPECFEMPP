// The only translation unit of specfem_compute that includes TensorOperations
// headers. It is excluded from unity batches in CMakeLists.txt.

#include "stiffness_graph.hpp"

#include <stdexcept>

#ifdef SPECFEM_ENABLE_TENSOROPS

#include "specfem/algorithms/gradient.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/subview.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>
#include <type_traits>

template <int NGLL, typename Tags>
void specfem::compute::impl::compute_stiffness_graph(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const specfem::datatype::ElementIndexRange &elements) {
  namespace tops = TensorOperations;
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  static_assert(
      specfem::element::attributes<Tags::dimension_tag,
                                   Tags::medium_tag>::components == 3,
      "the stiffness graph is written for three displacement components");

  if (elements.empty()) {
    return;
  }

  if (assembly.mesh.element_grid != NGLL) {
    throw std::runtime_error(
        "specfem::compute::compute_stiffness_graph: the number of GLL points "
        "in the mesh elements must match the template parameter NGLL.");
  }

  const auto &simulation_field =
      assembly.fields.template get_simulation_field<Tags::wavefield_tag>();
  // Medium-local DOF table of the range: element slot e is mesh element
  // elements.begin_index() + e.
  const auto field = specfem::assembly::subview(
      simulation_field,
      specfem::datatype::TaggedElementIndexRange<Tags>(elements));
  const auto s = field.get_field();
  const auto s_ddot = field.get_field_dot_dot();

  using PointTags =
      specfem::tags::Tags<Tags::dimension_tag, Tags::medium_tag,
                          Tags::property_tag, Tags::attenuation_tag, false>;
  using PointIndexType = specfem::point::index<Tags::dimension_tag, false>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<Tags::dimension_tag, true, false>;
  using PointPropertyType = specfem::point::properties<PointTags>;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<PointTags>;
  const auto jacobian_matrix = assembly.jacobian_matrix;
  const auto properties = assembly.properties;
  const auto weights = assembly.mesh.weights;
  const int begin_ispec = elements.begin_index();

  using Tiles =
      tops::LabelTiles<tops::LabelTile<'e', 1>, tops::LabelWhole<'g', NGLL>,
                       tops::LabelWhole<'b', NGLL>, tops::LabelWhole<'a', NGLL>,
                       tops::LabelWhole<'s', NGLL>>;

  auto g0 = tops::make_level_graph<type_real, ExecSpace>(Tiles{});

  auto [g1, lprime] = g0.add(tops::make_stage_node(tops::make_input_node(
      tops::make_handle<'a', 's'>(assembly.mesh.hprime))));

  auto [g2, idx] =
      g1.add(tops::make_index_node<'e', 'g', 'b', 'a'>(field.index_mapping));

  auto gather = [&](const int j) {
    return tops::make_gather_node<'e', 'g', 'b', 'a'>(
        idx, Kokkos::subview(s, Kokkos::ALL, j));
  };
  auto [g3, s0, s1, s2] = g2.add(gather(0), gather(1), gather(2));

  auto d_dxi = [&](const auto &s_j) {
    return tops::make_contraction_node<'e', 'g', 'b', 'a'>(
        lprime.template as<'a', 's'>(), s_j.template as<'e', 'g', 'b', 's'>());
  };
  auto d_deta = [&](const auto &s_j) {
    return tops::make_contraction_node<'e', 'g', 'b', 'a'>(
        lprime.template as<'b', 's'>(), s_j.template as<'e', 'g', 's', 'a'>());
  };
  auto d_dzeta = [&](const auto &s_j) {
    return tops::make_contraction_node<'e', 'g', 'b', 'a'>(
        lprime.template as<'g', 's'>(), s_j.template as<'e', 's', 'b', 'a'>());
  };
  auto [g4, ds0_dxi, ds1_dxi, ds2_dxi, ds0_deta, ds1_deta, ds2_deta, ds0_dzeta,
        ds1_dzeta, ds2_dzeta] =
      g3.add(d_dxi(s0), d_dxi(s1), d_dxi(s2), d_deta(s0), d_deta(s1),
             d_deta(s2), d_dzeta(s0), d_dzeta(s1), d_dzeta(s2));

  // F{i}{k} includes the quadrature weights and the Jacobian determinant.
  auto [g5, F00, F10, F20, F01, F11, F21, F02, F12, F22] =
      g4.add(tops::make_combine_node<'e', 'g', 'b', 'a'>(
          ds0_dxi, ds1_dxi, ds2_dxi, ds0_deta, ds1_deta, ds2_deta, ds0_dzeta,
          ds1_dzeta, ds2_dzeta,
          KOKKOS_LAMBDA(const int e, const int g, const int b, const int a,
                        const type_real ds0_dxi, const type_real ds1_dxi,
                        const type_real ds2_dxi, const type_real ds0_deta,
                        const type_real ds1_deta, const type_real ds2_deta,
                        const type_real ds0_dzeta, const type_real ds1_dzeta,
                        const type_real ds2_dzeta) {
            const PointIndexType index(begin_ispec + e, g, b, a);

            PointJacobianMatrixType point_jacobian_matrix;
            specfem::assembly::load_on_device(index, jacobian_matrix,
                                              point_jacobian_matrix);
            PointPropertyType point_property;
            specfem::assembly::load_on_device(index, properties,
                                              point_property);

            const type_real df_dxi[3] = { ds0_dxi, ds1_dxi, ds2_dxi };
            const type_real df_deta[3] = { ds0_deta, ds1_deta, ds2_deta };
            const type_real df_dgamma[3] = { ds0_dzeta, ds1_dzeta, ds2_dzeta };
            const auto du = specfem::algorithms::chain_rule(
                point_jacobian_matrix, df_dxi, df_deta, df_dgamma);

            const auto point_stress =
                specfem::medium_physics::compute_stress<PointTags>(
                    point_property, PointFieldDerivativesType(du));
            const auto F = point_stress * point_jacobian_matrix;

            const type_real w = weights(a) * weights(b) * weights(g);
            return Kokkos::Array{ w * F(0, 0), w * F(1, 0), w * F(2, 0),
                                  w * F(0, 1), w * F(1, 1), w * F(2, 1),
                                  w * F(0, 2), w * F(1, 2), w * F(2, 2) };
          }));

  auto div_xi = [&](const auto &F) {
    return tops::make_contraction_node<'e', 'g', 'b', 'a'>(
        lprime.template as<'s', 'a'>(), F.template as<'e', 'g', 'b', 's'>());
  };
  auto div_eta = [&](const auto &F) {
    return tops::make_contraction_node<'e', 'g', 'b', 'a'>(
        lprime.template as<'s', 'b'>(), F.template as<'e', 'g', 's', 'a'>());
  };
  auto div_zeta = [&](const auto &F) {
    return tops::make_contraction_node<'e', 'g', 'b', 'a'>(
        lprime.template as<'s', 'g'>(), F.template as<'e', 's', 'b', 'a'>());
  };
  auto [g6, Ks0_xi, Ks1_xi, Ks2_xi, Ks0_eta, Ks1_eta, Ks2_eta, Ks0_zeta,
        Ks1_zeta, Ks2_zeta] =
      g5.add(div_xi(F00), div_xi(F10), div_xi(F20), div_eta(F01), div_eta(F11),
             div_eta(F21), div_zeta(F02), div_zeta(F12), div_zeta(F22));

  auto [g7, r0, r1, r2] = g6.add(tops::make_combine_node<'e', 'g', 'b', 'a'>(
      Ks0_xi, Ks1_xi, Ks2_xi, Ks0_eta, Ks1_eta, Ks2_eta, Ks0_zeta, Ks1_zeta,
      Ks2_zeta,
      KOKKOS_LAMBDA(const int, const int, const int, const int,
                    const type_real Ks0_xi, const type_real Ks1_xi,
                    const type_real Ks2_xi, const type_real Ks0_eta,
                    const type_real Ks1_eta, const type_real Ks2_eta,
                    const type_real Ks0_zeta, const type_real Ks1_zeta,
                    const type_real Ks2_zeta) {
        return Kokkos::Array{ -(Ks0_xi + Ks0_eta + Ks0_zeta),
                              -(Ks1_xi + Ks1_eta + Ks1_zeta),
                              -(Ks2_xi + Ks2_eta + Ks2_zeta) };
      }));

  auto scatter = [&](const int i, const auto &r) {
    return tops::make_scatter_add_node<'e', 'g', 'b', 'a'>(
        idx, Kokkos::subview(s_ddot, Kokkos::ALL, i), r);
  };
  auto g8 = g7.add(scatter(0, r0), scatter(1, r1), scatter(2, r2));

  // No fence: see the header.
  g8.outputs().execute(tops::TeamPolicyTag<ExecSpace>{});
}

#else // !SPECFEM_ENABLE_TENSOROPS

template <int NGLL, typename Tags>
void specfem::compute::impl::compute_stiffness_graph(
    const specfem::assembly::assembly<Tags::dimension_tag> & /* assembly */,
    const specfem::datatype::ElementIndexRange & /* elements */) {
  throw std::runtime_error(
      "specfem::compute: the stiffness graph requires SPECFEM++ built with "
      "SPECFEM_ENABLE_TENSOROPS=ON (and SPECFEM_TENSOROPS_ROOT pointing at a "
      "TensorOperations checkout).");
}

#endif // SPECFEM_ENABLE_TENSOROPS

template void specfem::compute::impl::compute_stiffness_graph<
    5, specfem::compute::impl::stiffness_graph_tags>(
    const specfem::assembly::assembly<
        specfem::compute::impl::stiffness_graph_tags::dimension_tag> &,
    const specfem::datatype::ElementIndexRange &);
