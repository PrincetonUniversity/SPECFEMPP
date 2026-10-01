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
// actually builds with, the exact library features the tensor-graph stiffness
// kernel needs -- LevelGraph staged contractions on the TeamPolicyTag path,
// a contraction -> combine -> contraction chain across levels, a second
// blocked label (the kernel's identity-column axis), and a combine functor
// that reads a captured view at its GLOBAL output coordinate -- with no
// fixtures and no assembly. If a Kokkos or TensorOperations bump breaks the
// integration, this test fails first and in isolation.
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

  out.execute(tenops::TeamPolicyTag<ExecSpace>{}, C);
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

// A pointwise stage that captures a per-element view and reads it at the
// GLOBAL output coordinate, ignoring the column label -- the load-bearing
// combine semantics the stiffness kernel's constitutive functor stands on
// (material data does not depend on which column is pushed through).
struct ScalePointwise {
  Kokkos::View<type_real *[NGLL], Kokkos::LayoutRight,
               Kokkos::DefaultExecutionSpace>
      scale;

  KOKKOS_FUNCTION type_real operator()(const int e, const int /* column */,
                                       const int i, const type_real v) const {
    return scale(e, i) * v;
  }
};

// The structure the tensor-graph stiffness kernel issues, at toy extents: a
// contraction -> pointwise combine -> contraction chain over TWO blocked
// labels ('e' element, 'J' identity column),
//
//   grad(e,J,i) = sum_p A(i,p) U(e,J,p)
//   F(e,J,i)    = M(e,i) grad(e,J,i)        (combine, M read at global coords)
//   out(e,J,i)  = sum_p B(p,i) F(e,J,p)
//
// This is the proof that a second blocked label is accepted, that
// intermediates chain across levels, and that the combine functor receives
// global (not tile-local) coordinates -- in practice, not just by reading
// Evaluator/Team.hpp.
TEST(TensorOpsSmoke, ActionGraphPipelineMatchesHostLoop) {
  namespace tenops = TensorOperations;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  constexpr int ncolumns = 6;
  constexpr int column_tile = 2;
  using TileMap = tenops::LabelTiles<
      tenops::LabelTile<'e', 1>, tenops::LabelTile<'J', column_tile>,
      tenops::LabelWhole<'i', NGLL>, tenops::LabelWhole<'p', NGLL>,
      tenops::LabelWhole<'r', NGLL>>;

  Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, ExecSpace> A("A");
  Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, ExecSpace> B("B");
  Kokkos::View<type_real *[NGLL], Kokkos::LayoutRight, ExecSpace> M("M", nelem);
  Kokkos::View<type_real *[ncolumns][NGLL], Kokkos::LayoutRight, ExecSpace> U(
      "U", nelem);
  Kokkos::View<type_real *[ncolumns][NGLL], Kokkos::LayoutRight, ExecSpace> out(
      "out", nelem);

  auto h_A = Kokkos::create_mirror_view(A);
  auto h_B = Kokkos::create_mirror_view(B);
  auto h_M = Kokkos::create_mirror_view(M);
  auto h_U = Kokkos::create_mirror_view(U);
  std::size_t flat = 0;
  for (int i = 0; i < NGLL; ++i)
    for (int p = 0; p < NGLL; ++p) {
      h_A(i, p) = fill_value(flat++);
      h_B(i, p) = fill_value(flat++);
    }
  for (int e = 0; e < nelem; ++e)
    for (int i = 0; i < NGLL; ++i)
      h_M(e, i) = fill_value(flat++);
  for (int e = 0; e < nelem; ++e)
    for (int J = 0; J < ncolumns; ++J)
      for (int p = 0; p < NGLL; ++p)
        h_U(e, J, p) = fill_value(flat++);
  Kokkos::deep_copy(A, h_A);
  Kokkos::deep_copy(B, h_B);
  Kokkos::deep_copy(M, h_M);
  Kokkos::deep_copy(U, h_U);

  auto g0 = tenops::make_level_graph<type_real, ExecSpace>(TileMap{});
  auto [g1, a, b] =
      g0.add(tenops::make_stage_node(
                 tenops::make_input_node(tenops::make_handle<'r', 'p'>(A))),
             tenops::make_stage_node(
                 tenops::make_input_node(tenops::make_handle<'p', 'r'>(B))));
  auto [g2, u] = g1.add(tenops::make_stage_node(
      tenops::make_input_node(tenops::make_handle<'e', 'J', 'p'>(U))));
  auto [g3, grad] = g2.add(tenops::make_contraction_node<'e', 'J', 'i'>(
      a.template as<'i', 'p'>(), u));
  auto [g4, f] = g3.add(
      tenops::make_combine_node<'e', 'J', 'i'>(grad, ScalePointwise{ M }));
  auto [g5, result] = g4.add(tenops::make_contraction_node<'e', 'J', 'i'>(
      b.template as<'p', 'i'>(), f.template as<'e', 'J', 'p'>()));

  const auto graph_out = g5.outputs(result);
  const std::size_t scratch = graph_out.scratch_bytes();
  std::printf("[ scratch  ] action-graph pipeline: %zu bytes (cap %zu)\n",
              scratch, scratch_cap);
  EXPECT_LE(scratch, scratch_cap);

  graph_out.execute(tenops::TeamPolicyTag<ExecSpace>{}, out);
  Kokkos::fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  type_real max_abs_diff = 0;
  type_real max_abs_ref = 0;
  for (int e = 0; e < nelem; ++e)
    for (int J = 0; J < ncolumns; ++J)
      for (int i = 0; i < NGLL; ++i) {
        type_real ref = 0;
        for (int p = 0; p < NGLL; ++p) {
          type_real grad_ep = 0;
          for (int q = 0; q < NGLL; ++q)
            grad_ep += h_A(p, q) * h_U(e, J, q);
          ref += h_B(p, i) * h_M(e, p) * grad_ep;
        }
        max_abs_ref = std::max(max_abs_ref, std::abs(ref));
        max_abs_diff = std::max(max_abs_diff, std::abs(h_out(e, J, i) - ref));
      }
  EXPECT_LE(max_abs_diff, tolerance * std::max<type_real>(1, max_abs_ref))
      << "max |out_lib - out_ref| = " << max_abs_diff
      << " against max |out_ref| = " << max_abs_ref;
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
