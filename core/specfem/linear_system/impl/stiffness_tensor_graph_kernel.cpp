// The only translation unit that includes TensorOperations headers. Without
// SPECFEM_ENABLE_TENSOROPS the entry points are defined as throwing stubs
// (see the bottom of the file), so callers dispatch without preprocessor
// branches.
#include "specfem/linear_system/impl/stiffness_tensor_graph_kernel.hpp"

#ifdef SPECFEM_ENABLE_TENSOROPS

#include "specfem/algorithms.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/Tiling.hpp>

namespace specfem::linear_system_impl {

/**
 * @brief Pointwise constitutive stage of the stiffness tensor graph.
 *
 * Receives the nine reference-frame displacement gradients at one quadrature
 * point (three components by three reference directions), applies
 * `specfem::algorithms::chain_rule` (the same transform the gradient
 * algorithm closes with), delegates to
 * `specfem::medium_physics::compute_stress`, and returns the nine
 * stress-integrand values
 * \f$ F(c, r) = J \, \sigma_{cd} \, \partial r / \partial x_d \f$ grouped by
 * reference direction. The only medium-specific node in the graph.
 *
 * TensorOperations invokes the functor with the GLOBAL output coordinate
 * (element slot, column, iz, iy, ix); the column index is ignored because
 * metric and material data do not depend on which unit column is being
 * pushed through the action.
 */
template <int NGLL, typename Tags, typename JacobianMatrixType,
          typename PropertiesType>
struct StiffnessGraphIntegrand {
  using PointTags =
      specfem::tags::Tags<Tags::dimension_tag, Tags::medium_tag,
                          Tags::property_tag, Tags::attenuation_tag, false>;
  using PointIndexType = specfem::point::index<Tags::dimension_tag, false>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<Tags::dimension_tag, true, false>;
  using PointPropertyType = specfem::point::properties<PointTags>;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<PointTags>;

  JacobianMatrixType jacobian_matrix;
  PropertiesType properties;
  int batch_begin_ispec;

  KOKKOS_FUNCTION Kokkos::Array<type_real, 9>
  operator()(const int ielement, const int /* column */, const int iz,
             const int iy, const int ix, const type_real du0_dxi,
             const type_real du1_dxi, const type_real du2_dxi,
             const type_real du0_deta, const type_real du1_deta,
             const type_real du2_deta, const type_real du0_dgamma,
             const type_real du1_dgamma, const type_real du2_dgamma) const {
    const PointIndexType index(batch_begin_ispec + ielement, iz, iy, ix);

    PointJacobianMatrixType point_jacobian_matrix;
    specfem::assembly::load_on_device(index, jacobian_matrix,
                                      point_jacobian_matrix);

    PointPropertyType point_property;
    specfem::assembly::load_on_device(index, properties, point_property);

    const type_real du_dxi[3] = { du0_dxi, du1_dxi, du2_dxi };
    const type_real du_deta[3] = { du0_deta, du1_deta, du2_deta };
    const type_real du_dgamma[3] = { du0_dgamma, du1_dgamma, du2_dgamma };

    const PointFieldDerivativesType field_derivatives(
        specfem::algorithms::chain_rule(point_jacobian_matrix, du_dxi, du_deta,
                                        du_dgamma));
    const auto point_stress =
        specfem::medium_physics::compute_stress<PointTags>(point_property,
                                                           field_derivatives);
    // F(c, r) with the jacobian determinant folded in (stress.hpp operator*).
    const auto F = point_stress * point_jacobian_matrix;

    return { F(0, 0), F(1, 0), F(2, 0), F(0, 1), F(1, 1),
             F(2, 1), F(0, 2), F(1, 2), F(2, 2) };
  }
};

/**
 * @brief Transverse-weight sum closing the divergence, matching
 * `specfem::algorithms::impl::element_divergence`'s result stage (the summed
 * direction's weight already rides in the staged weighted derivative matrix).
 */
template <typename WeightsViewType> struct StiffnessGraphWeightedSum {
  WeightsViewType weights;

  KOKKOS_FUNCTION type_real operator()(const int /* ielement */,
                                       const int /* column */, const int iz,
                                       const int iy, const int ix,
                                       const type_real t_xi,
                                       const type_real t_eta,
                                       const type_real t_gamma) const {
    return weights(iz) * weights(iy) * t_xi +
           weights(iz) * weights(ix) * t_eta +
           weights(iy) * weights(ix) * t_gamma;
  }
};

} // namespace specfem::linear_system_impl

template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
void specfem::linear_system_impl::compute_element_stiffness_tensor_graph(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e) {

  constexpr auto dimension_tag = Tags::dimension_tag;
  constexpr auto medium_tag = Tags::medium_tag;
  constexpr int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;
  // The graph arity below (three staged component fields, nine gradients,
  // nine integrand outputs) is written for 3-component media; other media
  // swap the constitutive functor and the arity.
  static_assert(ncomp == 3, "tensor-graph kernel is written for 3-component "
                            "elastic media");
  constexpr int ndof = ncomp * NGLL * NGLL * NGLL;

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using JacobianMatrixType = std::decay_t<decltype(assembly.jacobian_matrix)>;
  using PropertiesType = std::decay_t<decltype(assembly.properties)>;

  if (batch.empty()) {
    return;
  }

  const int nbatch = batch.size();
  const int batch_begin_ispec = batch.begin_index();
  const auto hprime = assembly.mesh.hprime;
  const auto weights = assembly.mesh.weights;

  // Column tile: scratch scales linearly in the tile (~9.5 KB per column at
  // NGLL = 5, measured by tensorops_smoke_tests) and the tile must divide
  // ndof = 375. 15 fits the ~227 KB GPU opt-in team scratch, 3 the 32 KB
  // host-serial cap.
  constexpr bool on_gpu =
      !Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible;
  constexpr int column_tile = on_gpu ? 15 : 3;
  static_assert(ndof % column_tile == 0,
                "the column tile must divide the dof count");

  // The two 1D derivative operators the contractions stage: the gradient's
  // hprime(point, function) and the divergence's transposed
  // hprime(summed, point) with the summed point's quadrature weight folded in
  // (element_divergence keeps weights(l) inside the sum).
  Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, ExecSpace>
      derivative("specfem::linear_system::tensor_graph::derivative");
  Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, ExecSpace>
      weighted_transpose(
          "specfem::linear_system::tensor_graph::weighted_transpose");
  Kokkos::parallel_for(
      "specfem::linear_system::tensor_graph::stage_operators",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({ 0, 0 },
                                                        { NGLL, NGLL }),
      KOKKOS_LAMBDA(const int p, const int f) {
        derivative(p, f) = hprime(p, f);
        weighted_transpose(p, f) = hprime(p, f) * weights(p);
      });

  // Identity input: unit_columns(c, e, col, iz, iy, ix) = 1 exactly when col
  // is the local dof (c, iz, iy, ix). Constant data replicated over the
  // element axis because every graph operand must carry the labels of the
  // stage it feeds (broadcast labels are a TensorOperations follow-up).
  Kokkos::View<type_real ***[NGLL][NGLL][NGLL], Kokkos::LayoutRight, ExecSpace>
      unit_columns("specfem::linear_system::tensor_graph::unit_columns", ncomp,
                   nbatch, ndof);
  Kokkos::parallel_for(
      "specfem::linear_system::tensor_graph::fill_unit_columns",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
          { 0, 0, 0, 0, 0, 0 }, { ncomp, nbatch, ndof, NGLL, NGLL, NGLL }),
      KOKKOS_LAMBDA(const int c, const int e, const int col, const int iz,
                    const int iy, const int ix) {
        unit_columns(c, e, col, iz, iy, ix) =
            (col ==
             specfem::linear_system::local_dof_index<NGLL>(c, iz, iy, ix))
                ? static_cast<type_real>(1.0)
                : static_cast<type_real>(0.0);
      });

  // Graph output: forces(c, e, col, iz, iy, ix) = component c of the internal
  // force at (iz, iy, ix) from the unit displacement at local dof `col` --
  // i.e. K_e's column `col` before the reshape.
  Kokkos::View<type_real ***[NGLL][NGLL][NGLL], Kokkos::LayoutRight, ExecSpace>
      forces("specfem::linear_system::tensor_graph::forces", ncomp, nbatch,
             ndof);

  const auto all = Kokkos::ALL;
  const auto u0 = Kokkos::subview(unit_columns, 0, all, all, all, all, all);
  const auto u1 = Kokkos::subview(unit_columns, 1, all, all, all, all, all);
  const auto u2 = Kokkos::subview(unit_columns, 2, all, all, all, all, all);
  const auto f0 = Kokkos::subview(forces, 0, all, all, all, all, all);
  const auto f1 = Kokkos::subview(forces, 1, all, all, all, all, all);
  const auto f2 = Kokkos::subview(forces, 2, all, all, all, all, all);

  const specfem::linear_system_impl::StiffnessGraphIntegrand<
      NGLL, Tags, JacobianMatrixType, PropertiesType>
      integrand{ assembly.jacobian_matrix, assembly.properties,
                 batch_begin_ispec };
  const specfem::linear_system_impl::StiffnessGraphWeightedSum<
      std::decay_t<decltype(weights)>>
      weighted_sum{ weights };

  // Labels: e element slot, J identity column; k/j/i the point's z/y/x GLL
  // index; p the summed quadrature index; r the staged operators' point axis,
  // renamed per use to the axis each contraction reconstructs.
  namespace tenops = TensorOperations;
  using TileMap = tenops::LabelTiles<
      tenops::LabelTile<'e', 1>, tenops::LabelTile<'J', column_tile>,
      tenops::LabelWhole<'k', NGLL>, tenops::LabelWhole<'j', NGLL>,
      tenops::LabelWhole<'i', NGLL>, tenops::LabelWhole<'p', NGLL>,
      tenops::LabelWhole<'r', NGLL>>;

  auto g0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});
  auto [g1, h, hw] =
      g0.add(tenops::make_stage_node(tenops::make_input_node(
                 tenops::make_handle<'r', 'p'>(derivative))),
             tenops::make_stage_node(tenops::make_input_node(
                 tenops::make_handle<'p', 'r'>(weighted_transpose))));
  auto [g2, u0n, u1n, u2n] =
      g1.add(tenops::make_stage_node(tenops::make_input_node(
                 tenops::make_handle<'e', 'J', 'k', 'j', 'i'>(u0))),
             tenops::make_stage_node(tenops::make_input_node(
                 tenops::make_handle<'e', 'J', 'k', 'j', 'i'>(u1))),
             tenops::make_stage_node(tenops::make_input_node(
                 tenops::make_handle<'e', 'J', 'k', 'j', 'i'>(u2))));

  // Gradient level: du_c/dxi sums the x axis against hprime(ix, p), and so on
  // per direction (element_gradient's summation with the point index first).
  auto gradient_xi = [&](auto u) {
    return tenops::make_contraction_node<'e', 'J', 'k', 'j', 'i'>(
        h.template as<'i', 'p'>(), u.template as<'e', 'J', 'k', 'j', 'p'>());
  };
  auto gradient_eta = [&](auto u) {
    return tenops::make_contraction_node<'e', 'J', 'k', 'j', 'i'>(
        h.template as<'j', 'p'>(), u.template as<'e', 'J', 'k', 'p', 'i'>());
  };
  auto gradient_gamma = [&](auto u) {
    return tenops::make_contraction_node<'e', 'J', 'k', 'j', 'i'>(
        h.template as<'k', 'p'>(), u.template as<'e', 'J', 'p', 'j', 'i'>());
  };
  auto [g3, gxi0, gxi1, gxi2, geta0, geta1, geta2, ggamma0, ggamma1, ggamma2] =
      g2.add(gradient_xi(u0n), gradient_xi(u1n), gradient_xi(u2n),
             gradient_eta(u0n), gradient_eta(u1n), gradient_eta(u2n),
             gradient_gamma(u0n), gradient_gamma(u1n), gradient_gamma(u2n));

  // Constitutive level: one nine-output combine (chain rule + compute_stress
  // once per point). Operands ride bare -- their declared labels already are
  // the frame, and an explicit identity relabel could be mistyped silently
  // (every GLL axis has the same extent).
  auto [g4, fxi0, fxi1, fxi2, feta0, feta1, feta2, fgamma0, fgamma1, fgamma2] =
      g3.add(tenops::make_combine_node<'e', 'J', 'k', 'j', 'i'>(
          gxi0, gxi1, gxi2, geta0, geta1, geta2, ggamma0, ggamma1, ggamma2,
          integrand));

  // Divergence level: structurally the gradient level with the weighted
  // transposed operator (element_divergence's hprime(l, point) * weights(l)).
  auto divergence_xi = [&](auto f) {
    return tenops::make_contraction_node<'e', 'J', 'k', 'j', 'i'>(
        hw.template as<'p', 'i'>(), f.template as<'e', 'J', 'k', 'j', 'p'>());
  };
  auto divergence_eta = [&](auto f) {
    return tenops::make_contraction_node<'e', 'J', 'k', 'j', 'i'>(
        hw.template as<'p', 'j'>(), f.template as<'e', 'J', 'k', 'p', 'i'>());
  };
  auto divergence_gamma = [&](auto f) {
    return tenops::make_contraction_node<'e', 'J', 'k', 'j', 'i'>(
        hw.template as<'p', 'k'>(), f.template as<'e', 'J', 'p', 'j', 'i'>());
  };
  auto [g5, txi0, txi1, txi2, teta0, teta1, teta2, tgamma0, tgamma1, tgamma2] =
      g4.add(divergence_xi(fxi0), divergence_xi(fxi1), divergence_xi(fxi2),
             divergence_eta(feta0), divergence_eta(feta1),
             divergence_eta(feta2), divergence_gamma(fgamma0),
             divergence_gamma(fgamma1), divergence_gamma(fgamma2));

  auto weighted = [&](auto t_xi, auto t_eta, auto t_gamma) {
    return tenops::make_combine_node<'e', 'J', 'k', 'j', 'i'>(
        t_xi, t_eta, t_gamma, weighted_sum);
  };
  auto [g6, r0, r1, r2] =
      g5.add(weighted(txi0, teta0, tgamma0), weighted(txi1, teta1, tgamma1),
             weighted(txi2, teta2, tgamma2));

  // Instantiating the plan runs LevelGraph's structural guards (level
  // homogeneity, one iteration space per level, no member reading its own
  // level's output), which do not fire on their own.
  using Plan = tenops::LevelPlan<std::decay_t<decltype(g6.levels)>>;
  static_assert(Plan::num_levels == 6,
                "two stage levels and four compute levels");

  g6.outputs(r0, r1, r2)
      .execute(tenops::TeamPolicyTag2<ExecSpace>{}, f0, f1, f2);

  // Physics-free reshape into the k_e contract. Forward mapping only: the row
  // is local_dof_index of the force's (component, point), the column is the
  // identity column. Written with '=', no sign flip -- the divergence result
  // IS the internal force (K u = -accel before mass division), matching the
  // probe kernel.
  Kokkos::parallel_for(
      "specfem::linear_system::tensor_graph::reshape",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
          { 0, 0, 0, 0, 0, 0 }, { ncomp, nbatch, ndof, NGLL, NGLL, NGLL }),
      KOKKOS_LAMBDA(const int c, const int e, const int col, const int iz,
                    const int iy, const int ix) {
        k_e(e, specfem::linear_system::local_dof_index<NGLL>(c, iz, iy, ix),
            col) = forces(c, e, col, iz, iy, ix);
      });

  Kokkos::fence();
}

#else // !SPECFEM_ENABLE_TENSOROPS

#include <stdexcept>

// Throwing stub so OFF builds link and fail at the moment a tensor-graph
// kernel is actually requested, not at compile time in every caller.
template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
void specfem::linear_system_impl::compute_element_stiffness_tensor_graph(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const specfem::datatype::ElementIndexRange &,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &) {
  throw std::runtime_error(
      "specfem::linear_system::compute_element_stiffness: the tensor_graph "
      "kernel requires SPECFEM++ built with SPECFEM_ENABLE_TENSOROPS=ON "
      "(and SPECFEM_TENSOROPS_ROOT pointing at a TensorOperations "
      "checkout).");
}

#endif // SPECFEM_ENABLE_TENSOROPS

// Explicit instantiation: 3D elastic isotropic, NGLL = 5 (mirrors
// element_stiffness.cpp). Instantiates the real kernel or the throwing stub,
// whichever the build selected above.
template void
specfem::linear_system_impl::compute_element_stiffness_tensor_graph<
    5, specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                           specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::isotropic,
                           specfem::element::attenuation_tag::none>>(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const specfem::datatype::ElementIndexRange &,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &);
