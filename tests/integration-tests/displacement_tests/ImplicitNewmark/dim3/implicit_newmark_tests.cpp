#include "SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/solver.hpp"
#include "specfem/solver/implicit_solver.hpp"
#include "specfem/tags.hpp"
#include "specfem/timescheme.hpp"
#include "specfem/utilities/is_close.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace implicit_newmark_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr int ncomp = 3;

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using SolverTags =
    specfem::tags::Tags<dim3_tag, specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
using ImplicitSolverType = specfem::solver::ImplicitNewmarkSolver<SolverTags>;

// One time sample of one station/seismogram-type trace.
using Sample = std::array<type_real, ncomp>;
// Keyed by (station, network, seismogram type index).
using TraceMap = std::map<std::tuple<std::string, std::string, int>,
                          std::vector<std::pair<type_real, Sample>>>;

struct TestCase {
  std::unique_ptr<AssemblyType> assembly;
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme;
  specfem::runtime_configuration::setup setup;
};

// `test_name` names a fixture under this suite's data tree,
// displacement_tests/ImplicitNewmark/serial/dim3/ (mirroring the
// Newmark/serial/dim3 layout); paths are relative to TEST_OUTPUT_DIR.
TestCase build_case_3d(const std::string &test_name,
                       const std::string &config_name = "specfem_config.yaml") {
  const std::string test_path =
      "displacement_tests/ImplicitNewmark/serial/dim3/" + test_name;
  specfem::runtime_configuration::setup setup(test_path + "/" + config_name);

  const auto database_filename = setup.get_databases();
  const auto &source_entries = setup.get_source_entries();
  const auto stations_node = setup.get_stations();
  const auto quadratures = setup.instantiate_quadrature();

  auto mesh = specfem::io::read_3d_mesh(database_filename,
                                        setup.get_attenuation_setup());

  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  auto [sources, t0, starttime] = specfem::io::read_sources<dim3_tag>(
      source_entries, nsteps, setup.get_t0(), dt, setup.get_simulation_type());
  (void)starttime;
  setup.update_t0(t0);

  auto receivers = specfem::io::read_3d_receivers(stations_node);

  auto assembly = std::make_unique<AssemblyType>(
      mesh, quadratures, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, setup.get_max_seismogram_step(),
      setup.get_nstep_between_samples(), setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());
  auto time_scheme = setup.instantiate_timescheme(assembly->fields,
                                                  setup.get_simulation_type());
  return TestCase{ std::move(assembly), time_scheme, std::move(setup) };
}

// Collect all recorded seismograms of an assembly into a plain map.
TraceMap collect_traces(AssemblyType &assembly) {
  TraceMap traces;
  auto seismograms = assembly.receivers;
  seismograms.sync_seismograms();

  for (auto station_info : seismograms.stations()) {
    const std::string network_name = station_info.network_name;
    const std::string station_name = station_info.station_name;
    for (auto seismogram_type : station_info.get_seismogram_types()) {
      auto &trace = traces[{ station_name, network_name,
                             static_cast<int>(seismogram_type) }];
      for (auto [time, value] : seismograms.get_seismogram(
               station_name, network_name, seismogram_type)) {
        Sample sample{};
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          sample[icomp] = value[icomp];
        }
        trace.emplace_back(time, sample);
      }
    }
  }
  return traces;
}

// The implicit solver running the EXPLICIT scheme: acceleration form at
// beta = 0, gamma = 1/2 is the central-difference member of the Newmark
// family, so it must reproduce the production explicit solver step for step
// -- not to a calibrated tolerance, but as an algebraic identity.
//
// Why the two paths coincide exactly here. The explicit predictor forms
// u_pred and v_half = v_n + (dt/2) a_n, the force kernel computes
// f - K u_pred - C v_half, the result is divided by the effective mass
// M + (dt/2) lumped(C), and the corrector adds (dt/2) a_{n+1}. That IS the
// acceleration form at beta = 0, gamma = 1/2, whose operator is
// M + (dt/2) C. The one discrepancy is that production lumps C to a row-sum
// diagonal while DampingAssembler builds the full block-diagonal C -- so the
// identity holds only where C is empty. This fixture has no Stacey
// boundaries, which is exactly that case.
//
// This is THE gate on operator construction and time marching: a sign error
// in a coefficient, a mis-assembled K, or a mis-paired source-time function
// all show up here, and the assertion below names the first sample at which
// they do. It replaced two earlier tests that compared the beta = 1/4 run
// against the explicit solver -- those could only assert calibrated O(dt^2)
// dispersion bounds (0.15 / 0.35 / 0.55 relative L2), four orders of
// magnitude looser than what an exact scheme match can assert.
//
// Not covered here: the displacement form's time marching (the operator
// itself is pinned by the ImplicitSolver3D form-parameterized unit test).
// That form exists for the steady-state static solver, whose test will
// compare static displacement at chosen time steps -- through the
// seismogram interface, which is stable under field renumbering.
TEST(ImplicitNewmark3D, ReproducesExplicitSchemeAtBetaZero) {
  const std::string fixture = "HomogeneousHalfspaceSmallNoABCForceSource";

  TraceMap explicit_traces;
  {
    auto test_case = build_case_3d(fixture);
    auto solver = test_case.setup.instantiate_solver<5>(
        test_case.setup.get_dt(), *test_case.assembly, test_case.time_scheme,
        test_case.setup.get_simulation_type(), {});
    solver->run();
    explicit_traces = collect_traces(*test_case.assembly);
  }
  ASSERT_FALSE(explicit_traces.empty());

  TraceMap implicit_traces;
  {
    auto test_case = build_case_3d(fixture);
    specfem::solver::ImplicitSolverConfig config;
    config.form = specfem::solver::NewmarkForm::acceleration;
    config.newmark.beta = static_cast<type_real>(0);
    config.newmark.gamma = static_cast<type_real>(0.5);
    // At beta = 0 with C empty the operator is the diagonal lumped mass, so
    // RILUK(0) is an exact inverse and this costs one iteration per step.
    // Ask for more than the default anyway: the solve must not be what
    // limits the comparison below.
    config.gmres_tolerance =
        static_cast<specfem::linear_system::scalar_type>(1e-6);
    ImplicitSolverType solver(test_case.time_scheme, {}, *test_case.assembly,
                              config);
    solver.run();
    implicit_traces = collect_traces(*test_case.assembly);
  }

  ASSERT_EQ(implicit_traces.size(), explicit_traces.size());

  // The floor here is NOT the linear solve -- at beta = 0 with C empty the
  // operator is the diagonal lumped mass, RILUK(0) inverts it exactly, and
  // the solve is exact in one iteration. It is assembled K against
  // matrix-free K in single precision: production applies K matrix-free, the
  // implicit path applies the probed, assembled K. Same operator
  // mathematically, but ~2000 contributions per row summed in a different
  // order, and this build is float (SPECFEM_ENABLE_DOUBLE_PRECISION=OFF).
  //
  // Measured worst |implicit - explicit| / peak(explicit) on this fixture:
  // 3.8e-5, on acceleration late in the record (differentiation weights the
  // high frequencies, so the accumulated summation-order difference shows
  // there first). The tolerance leaves ~5x headroom, which the CUDA build
  // needs: a different backend sums each row in a different order again.
  //
  // If this ever fails just above the tolerance, that is the floor moving --
  // do not simply loosen it; check StiffnessAssembler3D first.
  constexpr double relative_tolerance = 2e-4;

  // Worst |implicit - explicit| / peak(explicit) over every sample, printed
  // on success so the headroom against the tolerance stays visible.
  double worst_deviation = 0;
  std::string worst_where;

  for (const auto &[key, explicit_trace] : explicit_traces) {
    const auto it = implicit_traces.find(key);
    ASSERT_NE(it, implicit_traces.end())
        << "station " << std::get<0>(key) << "." << std::get<1>(key)
        << " missing from the implicit run";
    const auto &implicit_trace = it->second;
    ASSERT_EQ(implicit_trace.size(), explicit_trace.size());

    // Scale the absolute floor to the trace, so the near-zero leading
    // samples (before the wavefront arrives) are not compared relatively.
    double peak = 0;
    for (const auto &[time, sample] : explicit_trace) {
      (void)time;
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        peak = std::max(peak, std::abs(static_cast<double>(sample[icomp])));
      }
    }
    ASSERT_GT(peak, 0.0) << "explicit trace is identically zero at station "
                         << std::get<0>(key) << "." << std::get<1>(key);
    const double absolute_tolerance = relative_tolerance * peak;

    // Walk forward in time and stop at the FIRST divergence: sample 0
    // implicates the right-hand side or the source-time pairing, a later
    // one implicates the state recovery. An aggregate norm would hide both.
    for (std::size_t sample = 0; sample < explicit_trace.size(); ++sample) {
      ASSERT_NEAR(implicit_trace[sample].first, explicit_trace[sample].first,
                  1e-3)
          << "seismogram time grids disagree";
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        const double expected =
            static_cast<double>(explicit_trace[sample].second[icomp]);
        const double actual =
            static_cast<double>(implicit_trace[sample].second[icomp]);
        const double deviation = std::abs(actual - expected) / peak;
        if (deviation > worst_deviation) {
          worst_deviation = deviation;
          worst_where = std::get<0>(key) + "." + std::get<1>(key) + " type " +
                        std::to_string(std::get<2>(key)) + " component " +
                        std::to_string(icomp) + " sample " +
                        std::to_string(sample);
        }
        ASSERT_TRUE(specfem::utilities::is_close(
            actual, expected, relative_tolerance, absolute_tolerance))
            << "implicit (beta = 0) diverged from the explicit solver at "
            << "sample " << sample << " (t = " << explicit_trace[sample].first
            << "), station " << std::get<0>(key) << "." << std::get<1>(key)
            << ", seismogram type " << std::get<2>(key) << ", component "
            << icomp << ": explicit " << expected << ", implicit " << actual
            << " (|difference| " << std::abs(actual - expected)
            << ", tolerance " << relative_tolerance << " relative / "
            << absolute_tolerance << " absolute)";
      }
    }
  }

  std::cout << "[   INFO   ] worst |implicit - explicit| / peak = "
            << worst_deviation << " at " << worst_where << " (tolerance "
            << relative_tolerance << ")" << std::endl;
}

} // namespace implicit_newmark_test

#else // !SPECFEM_ENABLE_TRILINOS

TEST(ImplicitNewmark3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM++ was built without Trilinos "
                  "(SPECFEM_ENABLE_TRILINOS=OFF); the implicit solver is "
                  "unavailable.";
}

#endif // SPECFEM_ENABLE_TRILINOS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
