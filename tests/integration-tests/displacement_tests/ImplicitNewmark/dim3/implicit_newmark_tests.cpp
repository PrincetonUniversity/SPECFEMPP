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
#include <array>
#include <cmath>
#include <cstddef>
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
  auto time_scheme = setup.instantiate_timescheme(assembly->fields);
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

// Implicit Newmark (beta = 1/4, gamma = 1/2) against the explicit solver on
// the same mesh, sources, and time grid. Both schemes are second-order but
// have different numerical dispersion, so agreement is O(dt^2)-bound, not
// roundoff-bound: the tolerance is calibrated, not derived. The fixture has
// no Stacey boundaries, so this exercises the M + K path (C empty).
TEST(ImplicitNewmark3D, MatchesExplicitRunOnNaturalBoundaryMesh) {
  const std::string fixture = "HomogeneousHalfspaceSmallNoABCForceSource";

  // Reference: the production explicit solver.
  TraceMap explicit_traces;
  {
    auto test_case = build_case_3d(fixture);
    auto solver = test_case.setup.instantiate_solver<5>(
        test_case.setup.get_dt(), *test_case.assembly, test_case.time_scheme,
        {});
    solver->run();
    explicit_traces = collect_traces(*test_case.assembly);
  }
  ASSERT_FALSE(explicit_traces.empty());

  // Implicit run on a fresh assembly (zeroed fields, fresh seismogram step).
  TraceMap implicit_traces;
  {
    auto test_case = build_case_3d(fixture);
    ImplicitSolverType solver(test_case.time_scheme, {}, *test_case.assembly);
    solver.run();
    implicit_traces = collect_traces(*test_case.assembly);
  }

  ASSERT_EQ(implicit_traces.size(), explicit_traces.size());

  // The two schemes carry opposite-sign period errors -- average
  // acceleration elongates by (w dt)^2 / 12, central difference shortens by
  // (w dt)^2 / 24 -- so at this fixture's coarse dt (0.035 s against a
  // 1.25 Hz Ricker with content to ~3 Hz) the phase gap accumulates to
  // ~1.6 rad at the top of the band over 75 steps: the implicit trace lags
  // the explicit one by up to a few samples late in the record. Measured
  // full-trace relative L2 gaps: ~0.10 (displacement), ~0.27 (velocity),
  // ~0.48 (acceleration) -- growing with derivative order because
  // differentiation weights the dispersive high frequencies. The tolerances
  // bound those calibrated values.
  const auto tolerance_for = [](const int type) -> double {
    switch (static_cast<specfem::enums::wavefield>(type)) {
    case specfem::enums::wavefield::displacement:
      return 0.15;
    case specfem::enums::wavefield::velocity:
      return 0.35;
    default:
      return 0.55;
    }
  };

  // No shift-alignment assertion on purpose: a +-1-sample comparison
  // between these two schemes is confounded in BOTH directions. Early in
  // the record the implicit trace leads by up to one sample (with beta > 0
  // the force at t_{n+1} moves u_{n+1} within the same step; with beta = 0
  // displacement cannot respond until the following step), while late in
  // the record the dispersion lag dominates in the other direction. Both
  // effects were measured on this fixture. Exact time-grid agreement is
  // asserted below via the recorded sample times; operator correctness is
  // pinned exactly by the ImplicitSolver3D.OperatorMatchesConstituents
  // unit test and the linear_system cross-path checks.
  for (const auto &[key, explicit_trace] : explicit_traces) {
    const auto it = implicit_traces.find(key);
    ASSERT_NE(it, implicit_traces.end())
        << "station " << std::get<0>(key) << "." << std::get<1>(key)
        << " missing from the implicit run";
    const auto &implicit_trace = it->second;
    ASSERT_EQ(implicit_trace.size(), explicit_trace.size());
    double error = 0;
    double reference_norm = 0;
    for (std::size_t sample = 0; sample < explicit_trace.size(); ++sample) {
      ASSERT_NEAR(implicit_trace[sample].first, explicit_trace[sample].first,
                  1e-3)
          << "seismogram time grids disagree";
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        const double difference =
            static_cast<double>(implicit_trace[sample].second[icomp]) -
            static_cast<double>(explicit_trace[sample].second[icomp]);
        const double reference =
            static_cast<double>(explicit_trace[sample].second[icomp]);
        error += difference * difference;
        reference_norm += reference * reference;
      }
    }
    ASSERT_GT(reference_norm, 0.0);

    const double relative_error = std::sqrt(error / reference_norm);
    EXPECT_LE(relative_error, tolerance_for(std::get<2>(key)))
        << "implicit vs explicit relative L2 mismatch at station "
        << std::get<0>(key) << "." << std::get<1>(key) << " (type "
        << std::get<2>(key) << ")";
  }
}

// Flattened (nglob x ncomp) copies of the final displacement and velocity
// fields of the forward elastic wavefield.
struct FinalFields {
  std::vector<double> displacement;
  std::vector<double> velocity;
};

FinalFields collect_final_fields(AssemblyType &assembly) {
  constexpr auto forward_tag = specfem::simulation::field_type::forward;
  constexpr auto elastic_tag = specfem::element::medium_tag::elastic;

  assembly.fields.copy_to_host();
  auto &field = assembly.fields.template get_simulation_field<forward_tag>();
  const auto &field_impl = field.template get_field<elastic_tag>();
  const auto h_u = field_impl.get_host_field();
  const auto h_v = field_impl.get_host_field_dot();
  const int nglob = field_impl.nglob;

  FinalFields fields;
  fields.displacement.reserve(static_cast<std::size_t>(nglob) * ncomp);
  fields.velocity.reserve(static_cast<std::size_t>(nglob) * ncomp);
  for (int iglob = 0; iglob < nglob; ++iglob) {
    for (int icomp = 0; icomp < ncomp; ++icomp) {
      fields.displacement.push_back(static_cast<double>(h_u(iglob, icomp)));
      fields.velocity.push_back(static_cast<double>(h_v(iglob, icomp)));
    }
  }
  return fields;
}

double relative_l2(const std::vector<double> &candidate,
                   const std::vector<double> &reference) {
  double error = 0;
  double reference_norm = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double difference = candidate[i] - reference[i];
    error += difference * difference;
    reference_norm += reference[i] * reference[i];
  }
  return std::sqrt(error / reference_norm);
}

// The static-solver test of issue #1984: a Heaviside step force on the
// Stacey-truncated halfspace, where the implicit solver taking 30 steps of
// dt = 0.7 with the dissipative Newmark preset must recreate the explicit
// solver taking 600 steps of dt = 0.035 to the same final time T = 21 s.
//
// Physics of the comparison: Stacey dashpots transmit no static load and K
// keeps its rigid null space, so the box does not converge to a fixed
// displacement -- it converges to a constant-velocity drift (the rigid part
// of C v balancing the net force) superposed on the converged elastic
// deformation. Hence:
//  - fields are compared at the SAME final time T (the drift contribution
//    to u grows linearly in time and would dominate any fixed-point
//    comparison),
//  - the steady-state detector is velocity/acceleration-based, and its
//    consistency is asserted on the velocity field only (v has converged to
//    the drift; u at an earlier stop time differs by drift x remaining
//    time).
// No analytic (Boussinesq-type) assertion on purpose: the dashpot-truncated
// box solves a different static boundary-value problem than the half space
// (traction-free rather than radiation-matched at the truncation faces), so
// near-source u_z ~ P / (16 pi mu (1 - nu) r) gives the order of magnitude
// only -- a human sanity check, not a tolerance one could defend.
TEST(ImplicitNewmark3D, RecreatesExplicitSteadyStateWithLargeSteps) {
  const std::string fixture = "HomogeneousHalfSpaceStaceyStatic";

  // Explicit reference: the "iterative time solver run to steady state".
  FinalFields explicit_fields;
  {
    auto test_case = build_case_3d(fixture, "specfem_config.yaml");
    auto solver = test_case.setup.instantiate_solver<5>(
        test_case.setup.get_dt(), *test_case.assembly, test_case.time_scheme,
        {});
    solver->run();
    explicit_fields = collect_final_fields(*test_case.assembly);
  }

  // Implicit static solve: 20x larger dissipative steps to the same T. One
  // solver instance serves both the full run and the detector rerun below
  // -- run() re-initializes the state, and reusing the instance skips a
  // second operator assembly + preconditioner setup (the dominant cost).
  specfem::solver::ImplicitSolverConfig config;
  config.newmark = specfem::solver::NewmarkBetaParameters::dissipative(
      static_cast<type_real>(0.6));
  auto test_case = build_case_3d(fixture, "specfem_config_implicit.yaml");
  ImplicitSolverType solver(test_case.time_scheme, {}, *test_case.assembly,
                            config);

  solver.run();
  EXPECT_EQ(solver.last_step(), test_case.setup.get_nsteps())
      << "with steady_state_tolerance = 0 the run must not stop early";
  const auto implicit_fields = collect_final_fields(*test_case.assembly);
  ASSERT_EQ(implicit_fields.displacement.size(),
            explicit_fields.displacement.size());

  // Tolerances calibrated on this fixture; the implicit large steps do not
  // resolve the band above ~1/(2 dt) = 0.7 Hz, but by T the transients have
  // been absorbed (explicit) or algorithmically damped (implicit), so the
  // surviving fields are the deformation + drift both schemes resolve.
  EXPECT_LE(
      relative_l2(implicit_fields.displacement, explicit_fields.displacement),
      0.05)
      << "final displacement field mismatch";
  EXPECT_LE(relative_l2(implicit_fields.velocity, explicit_fields.velocity),
            0.05)
      << "final velocity (drift) field mismatch";

  // Detector mechanism check only: at a coarse tolerance the stop must fire
  // well before the horizon (the ratios drop below 0.5 within a few steps of
  // the ramp ending). Where exactly a tight tolerance fires depends on the
  // fixture's physical ring-down rate -- deliberately not asserted; the
  // same-T comparison above is the sharp physics assertion. The solver logs
  // the per-step ratios when the detector is armed, for anyone tuning a
  // production tolerance.
  solver.set_steady_state_tolerance(static_cast<type_real>(0.5));
  solver.run();
  EXPECT_LT(solver.last_step(), test_case.setup.get_nsteps())
      << "steady-state detector did not fire at a coarse tolerance";
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
