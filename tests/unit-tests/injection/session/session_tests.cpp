// Tests for specfem::injection::ProviderSession, driving a mock plugin through
// the static-registration path: no mesh, no kernels, no shared library.

#include "mock_plugin.hpp"
#include "specfem/injection.hpp"
#include "specfem/injection/session.hpp"
#include "specfem/mpi.hpp"
#include <Kokkos_Core.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem::injection_session_test {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr int nstep = 10;
constexpr double tolerance = 1e-3;

// Window arithmetic is constexpr; pin it at compile time.
static_assert(specfem::injection::window_start(7, 4) == 4);
static_assert(specfem::injection::window_start(8, 4) == 8);
static_assert(specfem::injection::window_start(0, 3) == 0);
constexpr std::size_t bytes_per_row_step = 2 * sizeof(type_real);
static_assert(specfem::injection::window_steps_for_budget(
                  4 * 5 * 6 * bytes_per_row_step, 5, 6, nstep) == 4);
static_assert(specfem::injection::window_steps_for_budget(1, 5, 6, nstep) == 1);
static_assert(specfem::injection::window_steps_for_budget(1, 0, 6, nstep) ==
              nstep);
static_assert(specfem::injection::window_steps_for_budget(
                  std::size_t{ 1 } << 30, 5, 6, nstep) == nstep);

// Three elastic points (x = 1, 2, 3) followed by two acoustic ones (x = 4, 5).
specfem::injection::InjectionPoints test_points(const int nrows = 5) {
  specfem::injection::InjectionPoints points(nrows);
  for (int row = 0; row < nrows; ++row) {
    points.x(row) = static_cast<type_real>(row + 1);
    points.nz(row) = 1;
    points.medium(row) = row < 3 ? spfp_medium_elastic : spfp_medium_acoustic;
  }
  return points;
}

// Elastic: velocity + traction (6 slots); acoustic: potential_rate (1 slot).
specfem::injection::Request test_request() {
  return { { spfp_quantity_velocity, spfp_quantity_traction },
           { spfp_quantity_potential_rate } };
}

specfem::injection::SimulationParameters test_parameters() {
  return { -1.0, 0.5, nstep, { 0, 1, 0, 2, -3, 0 } };
}

specfem::injection::ProviderSession
make_session(const std::string &config,
             const specfem::injection::SessionOptions &options = {}) {
  return specfem::injection::ProviderSession(
      specfem::injection::PluginLibrary::open(mock_plugin_spec()), config,
      test_parameters(), options);
}

specfem::injection::SessionOptions window_of(const int steps) {
  specfem::injection::SessionOptions options;
  options.window_steps = steps;
  return options;
}

// Copy the current window back to the host through FrameBuffer::load, which
// runs on the device execution space.
Kokkos::View<type_real ***, Kokkos::HostSpace>
read_window(const specfem::injection::FrameBuffer &frame) {
  const int nrows = static_cast<int>(frame.data.extent(0));
  const int nslots = static_cast<int>(frame.data.extent(1));
  const int nsteps = frame.nsteps;
  const int start = frame.window_start;
  Kokkos::View<type_real ***, Kokkos::DefaultExecutionSpace> result(
      "result", nrows, nslots, nsteps);
  Kokkos::parallel_for(
      "read_window",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>(
          { 0, 0, 0 }, { nrows, nslots, nsteps }),
      KOKKOS_LAMBDA(const int row, const int slot, const int step) {
        result(row, slot, step) = frame.load(start + step, row, slot);
      });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  return host;
}

// Check the window against the mock's formula, for the slots each row uses.
void expect_window_matches(const specfem::injection::FrameBuffer &frame) {
  const auto values = read_window(frame);
  for (int row = 0; row < static_cast<int>(values.extent(0)); ++row) {
    const int slots = row < 3 ? 6 : 1;
    for (int slot = 0; slot < slots; ++slot)
      for (int step = 0; step < frame.nsteps; ++step)
        EXPECT_NEAR(values(row, slot, step),
                    mock_value(row + 1, slot, frame.window_start + step),
                    tolerance)
            << "row " << row << " slot " << slot << " step "
            << frame.window_start + step;
  }
}

std::string error_of(const std::function<void()> &call) {
  try {
    call();
  } catch (const std::exception &error) {
    return error.what();
  }
  return "<no error>";
}

TEST(ProviderSession, PassesSimulationInfoToCreate) {
  reset_mock_calls();
  auto session = make_session("ok");
  EXPECT_EQ(mock_calls().create, 1);
  EXPECT_EQ(mock_calls().info.dimension, 3);
  EXPECT_EQ(mock_calls().info.nstep, nstep);
  EXPECT_NEAR(mock_calls().info.t0, -1.0, tolerance);
  EXPECT_NEAR(mock_calls().info.dt, 0.5, tolerance);
  EXPECT_NEAR(mock_calls().info.bounds[4], -3.0, tolerance);
  EXPECT_EQ(mock_calls().communicator.size, specfem::MPI::get_size());
  EXPECT_EQ(mock_calls().communicator.rank, specfem::MPI::get_rank());
}

TEST(ProviderSession, WindowMatchesPluginValues) {
  reset_mock_calls();
  auto session = make_session("ok", window_of(4));
  session.setup(test_points(), test_request());
  EXPECT_EQ(mock_calls().setup_points, 5);

  session.ensure_window(0);
  EXPECT_EQ(session.frame().window_start, 0);
  EXPECT_EQ(session.frame().nsteps, 4);
  expect_window_matches(session.frame());
}

TEST(ProviderSession, WindowsAreAlignedAndLastIsPartial) {
  reset_mock_calls();
  auto session = make_session("ok", window_of(4));
  session.setup(test_points(), test_request());

  session.ensure_window(7);
  EXPECT_EQ(session.frame().window_start, 4);
  EXPECT_EQ(session.frame().nsteps, 4);

  session.ensure_window(9);
  EXPECT_EQ(session.frame().window_start, 8);
  EXPECT_EQ(session.frame().nsteps, 2);
  expect_window_matches(session.frame());
}

TEST(ProviderSession, ResultsDoNotDependOnOrder) {
  reset_mock_calls();
  auto session = make_session("ok", window_of(3));
  session.setup(test_points(), test_request());

  for (const int istep : { 9, 0, 5, 3 }) {
    session.ensure_window(istep);
    expect_window_matches(session.frame());
  }
  EXPECT_THAT(mock_calls().fill_starts, ElementsAre(9, 0, 3));

  // Steps inside the current window do not fill again.
  session.ensure_window(4);
  session.ensure_window(5);
  EXPECT_EQ(mock_calls().fill, 3);
}

TEST(ProviderSession, ZeroPointsStillRunsLifecycle) {
  reset_mock_calls();
  {
    auto session = make_session("ok");
    session.setup(test_points(0), test_request());
    EXPECT_EQ(session.window_steps(), nstep);
    session.ensure_window(0);
    EXPECT_EQ(mock_calls().setup_points, 0);
  }
  EXPECT_EQ(mock_calls().create, 1);
  EXPECT_EQ(mock_calls().setup, 1);
  EXPECT_EQ(mock_calls().fill, 1);
  EXPECT_EQ(mock_calls().destroy, 1);
}

TEST(ProviderSession, AutomaticWindowRespectsBudget) {
  reset_mock_calls();
  // 5 rows x 6 slots, staging + device.
  const std::size_t bytes_per_step = 2 * 5 * 6 * sizeof(type_real);
  specfem::injection::SessionOptions options;
  options.memory_budget = 4 * bytes_per_step + 1;
  auto session = make_session("ok", options);
  session.setup(test_points(), test_request());
  EXPECT_EQ(session.window_steps(), 4);

  // A budget below one step still gives one step; a large one caps at nstep.
  options.memory_budget = 1;
  auto tiny = make_session("ok", options);
  tiny.setup(test_points(), test_request());
  EXPECT_EQ(tiny.window_steps(), 1);

  options.memory_budget = std::size_t{ 1 } << 30;
  auto large = make_session("ok", options);
  large.setup(test_points(), test_request());
  EXPECT_EQ(large.window_steps(), nstep);
}

TEST(ProviderSession, ExplicitWindowOverridesBudget) {
  reset_mock_calls();
  specfem::injection::SessionOptions options = window_of(6);
  options.memory_budget = 1;
  auto session = make_session("ok", options);
  session.setup(test_points(), test_request());
  EXPECT_EQ(session.window_steps(), 6);

  auto invalid = make_session("ok", window_of(0));
  EXPECT_THAT(error_of([&] { invalid.setup(test_points(), test_request()); }),
              HasSubstr("window-steps must be at least 1"));
}

TEST(ProviderSession, CreateFailureCarriesPluginError) {
  reset_mock_calls();
  const auto error = error_of([] { make_session("fail-create"); });
  EXPECT_THAT(error, HasSubstr("'session-mock' create failed on rank"));
  EXPECT_THAT(error, HasSubstr("mock create failed"));
}

TEST(ProviderSession, SetupFailureCarriesPluginError) {
  reset_mock_calls();
  auto session = make_session("fail-setup");
  const auto error =
      error_of([&] { session.setup(test_points(), test_request()); });
  EXPECT_THAT(error, HasSubstr("'session-mock' setup failed on rank"));
  EXPECT_THAT(error, HasSubstr("mock setup failed"));
}

TEST(ProviderSession, UnsupportedQuantitiesAreAllListed) {
  reset_mock_calls();
  auto session = make_session("no-traction");
  const auto error =
      error_of([&] { session.setup(test_points(), test_request()); });
  EXPECT_THAT(error, HasSubstr("does not support elastic/traction, "
                               "acoustic/potential_rate"));
  // The plugin's setup is not called when the check fails.
  EXPECT_EQ(mock_calls().setup, 0);
}

TEST(ProviderSession, FillFailureCarriesPluginError) {
  reset_mock_calls();
  auto session = make_session("fail-fill", window_of(4));
  session.setup(test_points(), test_request());
  const auto error = error_of([&] { session.ensure_window(0); });
  EXPECT_THAT(error, HasSubstr("'session-mock' fill failed on rank"));
  EXPECT_THAT(error, HasSubstr("mock fill failed"));
}

TEST(ProviderSession, ValidationReportsNonFiniteValue) {
  reset_mock_calls();
  specfem::injection::SessionOptions options = window_of(4);
  options.validate = true;
  auto session = make_session("nan-fill", options);
  session.setup(test_points(), test_request());
  session.ensure_window(0); // the NaN is in step 5
  const auto error = error_of([&] { session.ensure_window(5); });
  EXPECT_THAT(error, HasSubstr("non-finite value"));
  EXPECT_THAT(error, HasSubstr("at point 1, slot 2, step 5"));

  options.validate = false;
  auto unchecked = make_session("nan-fill", options);
  unchecked.setup(test_points(), test_request());
  EXPECT_NO_THROW(unchecked.ensure_window(5));
}

TEST(ProviderSession, RejectsStepsOutsideSimulation) {
  reset_mock_calls();
  auto session = make_session("ok", window_of(4));
  EXPECT_THROW(session.ensure_window(0), std::logic_error);
  session.setup(test_points(), test_request());
  EXPECT_THROW(session.ensure_window(-1), std::out_of_range);
  EXPECT_THROW(session.ensure_window(nstep), std::out_of_range);
}

} // namespace specfem::injection_session_test
