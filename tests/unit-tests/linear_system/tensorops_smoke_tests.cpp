#include "../SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TENSOROPS

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/Tiling.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>

// Compile-and-compare spike for the TensorOperations dependency (issue #2066),
// kept as a permanent smoke test. It proves, against the Kokkos SPECFEM++
// actually builds with, the exact library features the sum-factored stiffness
// kernel needs -- a LevelGraph staged contraction on the TeamPolicyTag2 path,
// including the production-shaped rank-7 output -- before any kernel code
// exists. If a Kokkos or TensorOperations bump breaks the integration, this
// test fails first and in isolation.
namespace tensorops_smoke_test {

constexpr int NGLL = 5;
constexpr int ncomp = 3;
constexpr int nelem = 4;

constexpr bool single_precision = sizeof(type_real) == sizeof(float);
constexpr type_real tolerance = single_precision ? 1e-5 : 1e-12;

// Team scratch a level-0 request may not exceed: Kokkos' host-serial cap is
// 32 KB; on GPU the opt-in shared-memory ceiling is far higher (227 KB on
// H100) and anything this test builds is nowhere near it.
constexpr bool is_gpu =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                Kokkos::HostSpace>::accessible;
constexpr std::size_t scratch_cap = is_gpu ? 227u * 1024u : 32u * 1024u;

// Deterministic, sign-alternating fill so sums neither vanish nor grow.
type_real fill_value(std::size_t flat_index) {
  return static_cast<type_real>(
             static_cast<double>((flat_index * 31 + 7) % 13) - 6.0) /
         static_cast<type_real>(7.0);
}

// The smallest graph with the production structure: two staged inputs on
// separate levels (their iteration spaces differ, and a level's members must
// share one), then one contraction summing the shared mode.
//
//   C(e,i) = sum_p A(i,p) * U(e,p)
TEST(TensorOpsSmoke, ToyContractionMatchesHostLoop) {
  namespace tenops = TensorOperations;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using TileMap = tenops::LabelTiles<tenops::LabelTile<'e', 1>,
                                     tenops::LabelWhole<'i', NGLL>,
                                     tenops::LabelWhole<'p', NGLL>>;

  Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, ExecSpace> A("A");
  Kokkos::View<type_real *[NGLL], Kokkos::LayoutRight, ExecSpace> U("U", nelem);
  Kokkos::View<type_real *[NGLL], Kokkos::LayoutRight, ExecSpace> C("C", nelem);

  auto h_A = Kokkos::create_mirror_view(A);
  auto h_U = Kokkos::create_mirror_view(U);
  for (int i = 0; i < NGLL; ++i)
    for (int p = 0; p < NGLL; ++p)
      h_A(i, p) = fill_value(static_cast<std::size_t>(i * NGLL + p));
  for (int e = 0; e < nelem; ++e)
    for (int p = 0; p < NGLL; ++p)
      h_U(e, p) = fill_value(static_cast<std::size_t>(1000 + e * NGLL + p));
  Kokkos::deep_copy(A, h_A);
  Kokkos::deep_copy(U, h_U);

  auto g0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});
  auto [g1, a] = g0.add(tenops::make_stage_node(
      tenops::make_input_node(tenops::make_handle<'i', 'p'>(A))));
  auto [g2, u] = g1.add(tenops::make_stage_node(
      tenops::make_input_node(tenops::make_handle<'e', 'p'>(U))));
  auto [g3, c] = g2.add(tenops::make_contraction_node<'e', 'i'>(a, u));

  const auto out = g3.outputs(c);
  const std::size_t scratch = out.scratch_bytes();
  std::printf("[ scratch  ] toy contraction graph: %zu bytes\n", scratch);
  EXPECT_LE(scratch, scratch_cap);

  out.execute(tenops::TeamPolicyTag2<ExecSpace>{}, C);
  Kokkos::fence();

  auto h_C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  type_real max_ref = 0;
  for (int e = 0; e < nelem; ++e)
    for (int i = 0; i < NGLL; ++i) {
      type_real ref = 0;
      for (int p = 0; p < NGLL; ++p)
        ref += h_A(i, p) * h_U(e, p);
      max_ref = std::max(max_ref, std::abs(ref));
      EXPECT_NEAR(h_C(e, i), ref, tolerance * std::max<type_real>(1, max_ref))
          << "at (e=" << e << ", i=" << i << ")";
    }
}

// The exact contraction shape the sum-factored stiffness kernel issues for
// each of its three diagonal (r == s) blocks: the rank-7 per-element block
//
//   T(e,c,b,k,j,i,x) = sum_p HH(p,i,x) * G(e,c,b,k,j,p)
//
// with HH(p,i,x) = h'_i(xi_p) h'_x(xi_p) and G the weighted material-metric
// tensor. Every output mode appears in exactly one operand and the single
// shared mode 'p' is summed, which is what make_contraction_node requires;
// this test is the proof that a rank-7 declared output and a 3-extent
// LabelWhole are accepted in practice, not just by reading NodeHandle.hpp.
TEST(TensorOpsSmoke, ProductionShapedRank7ContractionMatchesHostLoop) {
  namespace tenops = TensorOperations;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using TileMap = tenops::LabelTiles<
      tenops::LabelTile<'e', 1>, tenops::LabelWhole<'c', ncomp>,
      tenops::LabelWhole<'b', ncomp>, tenops::LabelWhole<'k', NGLL>,
      tenops::LabelWhole<'j', NGLL>, tenops::LabelWhole<'i', NGLL>,
      tenops::LabelWhole<'x', NGLL>, tenops::LabelWhole<'p', NGLL>>;

  Kokkos::View<type_real[NGLL][NGLL][NGLL], Kokkos::LayoutRight, ExecSpace> HH(
      "HH");
  Kokkos::View<type_real *[ncomp][ncomp][NGLL][NGLL][NGLL], Kokkos::LayoutRight,
               ExecSpace>
      G("G", nelem);
  Kokkos::View<type_real *[ncomp][ncomp][NGLL][NGLL][NGLL][NGLL],
               Kokkos::LayoutRight, ExecSpace>
      T("T", nelem);

  auto h_HH = Kokkos::create_mirror_view(HH);
  auto h_G = Kokkos::create_mirror_view(G);
  std::size_t flat = 0;
  for (int p = 0; p < NGLL; ++p)
    for (int i = 0; i < NGLL; ++i)
      for (int x = 0; x < NGLL; ++x)
        h_HH(p, i, x) = fill_value(flat++);
  flat = 5000;
  for (int e = 0; e < nelem; ++e)
    for (int c = 0; c < ncomp; ++c)
      for (int b = 0; b < ncomp; ++b)
        for (int k = 0; k < NGLL; ++k)
          for (int j = 0; j < NGLL; ++j)
            for (int p = 0; p < NGLL; ++p)
              h_G(e, c, b, k, j, p) = fill_value(flat++);
  Kokkos::deep_copy(HH, h_HH);
  Kokkos::deep_copy(G, h_G);

  auto g0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});
  auto [g1, hh] = g0.add(tenops::make_stage_node(
      tenops::make_input_node(tenops::make_handle<'p', 'i', 'x'>(HH))));
  auto [g2, g] = g1.add(tenops::make_stage_node(tenops::make_input_node(
      tenops::make_handle<'e', 'c', 'b', 'k', 'j', 'p'>(G))));
  auto [g3, t] = g2.add(
      tenops::make_contraction_node<'e', 'c', 'b', 'k', 'j', 'i', 'x'>(hh, g));

  const auto out = g3.outputs(t);
  const std::size_t scratch = out.scratch_bytes();
  std::printf("[ scratch  ] rank-7 diagonal-block graph: %zu bytes "
              "(cap %zu)\n",
              scratch, scratch_cap);
  EXPECT_LE(scratch, scratch_cap);

  out.execute(tenops::TeamPolicyTag2<ExecSpace>{}, T);
  Kokkos::fence();

  auto h_T = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, T);
  type_real max_abs_diff = 0;
  type_real max_abs_ref = 0;
  for (int e = 0; e < nelem; ++e)
    for (int c = 0; c < ncomp; ++c)
      for (int b = 0; b < ncomp; ++b)
        for (int k = 0; k < NGLL; ++k)
          for (int j = 0; j < NGLL; ++j)
            for (int i = 0; i < NGLL; ++i)
              for (int x = 0; x < NGLL; ++x) {
                type_real ref = 0;
                for (int p = 0; p < NGLL; ++p)
                  ref += h_HH(p, i, x) * h_G(e, c, b, k, j, p);
                max_abs_ref = std::max(max_abs_ref, std::abs(ref));
                max_abs_diff = std::max(
                    max_abs_diff, std::abs(h_T(e, c, b, k, j, i, x) - ref));
              }
  EXPECT_LE(max_abs_diff, tolerance * std::max<type_real>(1, max_abs_ref))
      << "max |T_lib - T_ref| = " << max_abs_diff
      << " against max |T_ref| = " << max_abs_ref;
}

} // namespace tensorops_smoke_test

#else // !SPECFEM_ENABLE_TENSOROPS

TEST(TensorOpsSmoke, SkippedWithoutTensorOps) {
  GTEST_SKIP() << "SPECFEM++ was built without TensorOperations "
                  "(SPECFEM_ENABLE_TENSOROPS=OFF); the smoke test is "
                  "unavailable.";
}

#endif // SPECFEM_ENABLE_TENSOROPS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
