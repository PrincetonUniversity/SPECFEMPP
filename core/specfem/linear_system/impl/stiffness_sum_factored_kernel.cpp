// The only translation unit that includes TensorOperations headers; exists
// solely in SPECFEM_ENABLE_TENSOROPS builds (see the header).
#ifdef SPECFEM_ENABLE_TENSOROPS

#include "specfem/linear_system/impl/stiffness_sum_factored_kernel.hpp"

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/Tiling.hpp>

template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::attenuation_tag == specfem::element::attenuation_tag::none)
void specfem::linear_system_impl::compute_element_stiffness_sum_factored(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e) {

  constexpr auto dimension_tag = Tags::dimension_tag;
  constexpr auto medium_tag = Tags::medium_tag;
  constexpr int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;
  // The G_rs assembly below hard-codes the 3x3 elastic constitutive form.
  static_assert(ncomp == 3, "sum-factored kernel is written for 3-component "
                            "elastic media");
  constexpr int ndof = ncomp * NGLL * NGLL * NGLL;

  constexpr bool using_simd = false;
  using PointTags =
      specfem::tags::Tags<dimension_tag, medium_tag, Tags::property_tag,
                          Tags::attenuation_tag, using_simd>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension_tag, true, using_simd>;
  using PointPropertyType = specfem::point::properties<PointTags>;
  using PointIndexType = specfem::point::index<dimension_tag, using_simd>;
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  if (batch.empty()) {
    return;
  }

  const int nbatch = batch.size();
  const int batch_begin_ispec = batch.begin_index();
  const auto hprime = assembly.mesh.hprime;
  const auto weights = assembly.mesh.weights;
  const auto &jacobian_matrix = assembly.jacobian_matrix;
  const auto &properties = assembly.properties;

  // Lagrange derivative outer product HH(p, a, b) = h'_a(xi_p) h'_b(xi_p):
  // the 1D quadrature-line factor shared by the three diagonal contractions.
  Kokkos::View<type_real[NGLL][NGLL][NGLL], Kokkos::LayoutRight, ExecSpace> HH(
      "specfem::linear_system::sum_factored::HH");
  Kokkos::parallel_for(
      "specfem::linear_system::sum_factored::hh_outer_product",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({ 0, 0, 0 },
                                                        { NGLL, NGLL, NGLL }),
      KOKKOS_LAMBDA(const int p, const int a, const int b) {
        HH(p, a, b) = hprime(p, a) * hprime(p, b);
      });

  // Weighted material-metric tensor, indexed (r, s, e, c, b, iz, iy, ix):
  //   G_rs^{cb}(q) = w(q) J(q) [ lambda xi_{r,c} xi_{s,b}
  //                              + mu xi_{r,b} xi_{s,c}
  //                              + mu delta_{cb} sum_a xi_{r,a} xi_{s,a} ]
  // with xi_{r,a} the jacobian-matrix entry d(r)/d(x_a). The reference
  // directions lead the layout so each (r, s) slice is a LayoutRight rank-6
  // view TensorOperations can stage directly.
  Kokkos::View<type_real ***[3][3][NGLL][NGLL][NGLL], Kokkos::LayoutRight,
               ExecSpace>
      G("specfem::linear_system::sum_factored::G", 3, 3, nbatch);
  Kokkos::parallel_for(
      "specfem::linear_system::sum_factored::material_metric",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
          { 0, 0, 0, 0 }, { nbatch, NGLL, NGLL, NGLL }),
      KOKKOS_LAMBDA(const int ielement, const int iz, const int iy,
                    const int ix) {
        const PointIndexType index(batch_begin_ispec + ielement, iz, iy, ix);

        PointJacobianMatrixType point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);

        PointPropertyType point_property;
        specfem::assembly::load_on_device(index, properties, point_property);

        const type_real xi[3][3] = {
          { point_jacobian_matrix.xix, point_jacobian_matrix.xiy,
            point_jacobian_matrix.xiz },
          { point_jacobian_matrix.etax, point_jacobian_matrix.etay,
            point_jacobian_matrix.etaz },
          { point_jacobian_matrix.gammax, point_jacobian_matrix.gammay,
            point_jacobian_matrix.gammaz }
        };
        const type_real lambda = point_property.lambda();
        const type_real mu = point_property.mu();
        const type_real wj = weights(iz) * weights(iy) * weights(ix) *
                             point_jacobian_matrix.jacobian;

        for (int r = 0; r < 3; ++r) {
          for (int s = 0; s < 3; ++s) {
            type_real metric_dot = 0;
            for (int a = 0; a < 3; ++a) {
              metric_dot += xi[r][a] * xi[s][a];
            }
            for (int c = 0; c < ncomp; ++c) {
              for (int b = 0; b < ncomp; ++b) {
                type_real value =
                    lambda * xi[r][c] * xi[s][b] + mu * xi[r][b] * xi[s][c];
                if (c == b) {
                  value += mu * metric_dot;
                }
                G(r, s, ielement, c, b, iz, iy, ix) = wj * value;
              }
            }
          }
        }
      });

  // Diagonal (r == s) blocks: true contractions over the shared quadrature
  // line, one LevelGraph execute per reference direction (fusing all three
  // would triple the scratch request past the 32 KB host-serial cap; each
  // graph asks for ~27.5 KB at tile extent 1 -- see tensorops_smoke_tests).
  // The two stages sit on separate levels because their iteration spaces
  // differ (125 points against 1125 per element tile).
  //
  // Labels: e element, c row component, b column component; k/j/i the row
  // dof's z/y/x GLL index, z/y/x the column dof's; p the summed quadrature
  // line. Cross (r != s) blocks are NOT expressible as contractions here --
  // every operand mode would have to survive to the output -- so they stay
  // pointwise products in the epilogue below.
  namespace tenops = TensorOperations;
  using TileMap = tenops::LabelTiles<
      tenops::LabelTile<'e', 1>, tenops::LabelWhole<'c', ncomp>,
      tenops::LabelWhole<'b', ncomp>, tenops::LabelWhole<'k', NGLL>,
      tenops::LabelWhole<'j', NGLL>, tenops::LabelWhole<'i', NGLL>,
      tenops::LabelWhole<'x', NGLL>, tenops::LabelWhole<'y', NGLL>,
      tenops::LabelWhole<'z', NGLL>, tenops::LabelWhole<'p', NGLL>>;
  using DiagonalBlockView =
      Kokkos::View<type_real *[ncomp][ncomp][NGLL][NGLL][NGLL][NGLL],
                   Kokkos::LayoutRight, ExecSpace>;

  const auto all = Kokkos::ALL;
  const auto g_xixi = Kokkos::subview(G, 0, 0, all, all, all, all, all, all);
  const auto g_etaeta = Kokkos::subview(G, 1, 1, all, all, all, all, all, all);
  const auto g_gammagamma =
      Kokkos::subview(G, 2, 2, all, all, all, all, all, all);

  // T_xx(e,c,b,iz,iy,ix,jx) = sum_p HH(p,ix,jx) G_xixi(e,c,b,iz,iy,p)
  DiagonalBlockView T_xx("specfem::linear_system::sum_factored::T_xx", nbatch);
  {
    auto graph0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});
    auto [graph1, hh] = graph0.add(tenops::make_stage_node(
        tenops::make_input_node(tenops::make_handle<'p', 'i', 'x'>(HH))));
    auto [graph2, g] =
        graph1.add(tenops::make_stage_node(tenops::make_input_node(
            tenops::make_handle<'e', 'c', 'b', 'k', 'j', 'p'>(g_xixi))));
    auto [graph3, t] = graph2.add(
        tenops::make_contraction_node<'e', 'c', 'b', 'k', 'j', 'i', 'x'>(hh,
                                                                         g));
    graph3.outputs(t).execute(tenops::TeamPolicyTag2<ExecSpace>{}, T_xx);
  }

  // T_yy(e,c,b,iz,ix,iy,jy) = sum_p HH(p,iy,jy) G_etaeta(e,c,b,iz,p,ix)
  DiagonalBlockView T_yy("specfem::linear_system::sum_factored::T_yy", nbatch);
  {
    auto graph0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});
    auto [graph1, hh] = graph0.add(tenops::make_stage_node(
        tenops::make_input_node(tenops::make_handle<'p', 'j', 'y'>(HH))));
    auto [graph2, g] =
        graph1.add(tenops::make_stage_node(tenops::make_input_node(
            tenops::make_handle<'e', 'c', 'b', 'k', 'p', 'i'>(g_etaeta))));
    auto [graph3, t] = graph2.add(
        tenops::make_contraction_node<'e', 'c', 'b', 'k', 'i', 'j', 'y'>(hh,
                                                                         g));
    graph3.outputs(t).execute(tenops::TeamPolicyTag2<ExecSpace>{}, T_yy);
  }

  // T_zz(e,c,b,iy,ix,iz,jz) = sum_p HH(p,iz,jz) G_gammagamma(e,c,b,p,iy,ix)
  DiagonalBlockView T_zz("specfem::linear_system::sum_factored::T_zz", nbatch);
  {
    auto graph0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});
    auto [graph1, hh] = graph0.add(tenops::make_stage_node(
        tenops::make_input_node(tenops::make_handle<'p', 'k', 'z'>(HH))));
    auto [graph2, g] =
        graph1.add(tenops::make_stage_node(tenops::make_input_node(
            tenops::make_handle<'e', 'c', 'b', 'p', 'j', 'i'>(g_gammagamma))));
    auto [graph3, t] = graph2.add(
        tenops::make_contraction_node<'e', 'c', 'b', 'j', 'i', 'k', 'z'>(hh,
                                                                         g));
    graph3.outputs(t).execute(tenops::TeamPolicyTag2<ExecSpace>{}, T_zz);
  }

  // Epilogue: one flat pass over the K_e entries. GLL collocation pins the
  // quadrature point of every cross (r != s) term -- the r-coordinate to the
  // column dof's, the s-coordinate to the row dof's, the shared coordinate to
  // both (the delta) -- so each surviving term is a single product. The three
  // diagonal contractions enter under their transverse deltas. This write
  // owns the dof ordering (specfem::linear_system::local_dof_index) and the
  // sign convention (the sum IS the internal force: K u = -accel, written
  // with '=', no flip), matching the probe kernel.
  Kokkos::parallel_for(
      "specfem::linear_system::sum_factored::epilogue",
      Kokkos::RangePolicy<ExecSpace>(0, nbatch * ndof * ndof),
      KOKKOS_LAMBDA(const int flat) {
        constexpr int points = NGLL * NGLL * NGLL;
        const int e = flat / (ndof * ndof);
        const int entry = flat % (ndof * ndof);
        const int row = entry / ndof;
        const int col = entry % ndof;
        // Inverse of local_dof_index: dof = c * NGLL^3 + (iz*NGLL + iy)*NGLL +
        // ix
        const int c = row / points;
        const int row_point = row % points;
        const int iz = row_point / (NGLL * NGLL);
        const int iy = (row_point / NGLL) % NGLL;
        const int ix = row_point % NGLL;
        const int b = col / points;
        const int col_point = col % points;
        const int jz = col_point / (NGLL * NGLL);
        const int jy = (col_point / NGLL) % NGLL;
        const int jx = col_point % NGLL;

        type_real value = 0;
        if (iz == jz && iy == jy) {
          value += T_xx(e, c, b, iz, iy, ix, jx);
        }
        if (iz == jz && ix == jx) {
          value += T_yy(e, c, b, iz, ix, iy, jy);
        }
        if (iy == jy && ix == jx) {
          value += T_zz(e, c, b, iy, ix, iz, jz);
        }
        if (iz == jz) {
          value += hprime(jx, ix) * G(0, 1, e, c, b, iz, iy, jx) *
                   hprime(iy, jy); // xi-eta
          value += hprime(jy, iy) * G(1, 0, e, c, b, iz, jy, ix) *
                   hprime(ix, jx); // eta-xi
        }
        if (iy == jy) {
          value += hprime(jx, ix) * G(0, 2, e, c, b, iz, iy, jx) *
                   hprime(iz, jz); // xi-gamma
          value += hprime(jz, iz) * G(2, 0, e, c, b, jz, iy, ix) *
                   hprime(ix, jx); // gamma-xi
        }
        if (ix == jx) {
          value += hprime(jy, iy) * G(1, 2, e, c, b, iz, jy, ix) *
                   hprime(iz, jz); // eta-gamma
          value += hprime(jz, iz) * G(2, 1, e, c, b, jz, iy, ix) *
                   hprime(iy, jy); // gamma-eta
        }
        k_e(e, row, col) = value;
      });

  Kokkos::fence();
}

// Explicit instantiation: 3D elastic isotropic, NGLL = 5 (mirrors
// element_stiffness.cpp).
template void
specfem::linear_system_impl::compute_element_stiffness_sum_factored<
    5, specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                           specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::isotropic,
                           specfem::element::attenuation_tag::none>>(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const specfem::datatype::ElementIndexRange &,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &);

#endif // SPECFEM_ENABLE_TENSOROPS
