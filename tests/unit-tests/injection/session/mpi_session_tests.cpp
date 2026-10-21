// Multi-rank tests for specfem::injection::ProviderSession: ranks with
// different numbers of points (including none) agree on the window size and
// on failures.

#include "mock_plugin.hpp"
#include "specfem/injection.hpp"
#include "specfem/injection/session.hpp"
#include "specfem/mpi.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace specfem::injection_session_test {

using ::testing::HasSubstr;

constexpr int nstep = 10;

// Rank 0 holds four elastic points; every other rank holds none.
specfem::injection::InjectionPoints rank_points() {
  const int nrows = specfem::MPI::get_rank() == 0 ? 4 : 0;
  specfem::injection::InjectionPoints points(nrows);
  for (int row = 0; row < nrows; ++row) {
    points.x(row) = static_cast<type_real>(row);
    points.medium(row) = spfp_medium_elastic;
  }
  return points;
}

specfem::injection::ProviderSession
make_session(const std::string &config,
             const specfem::injection::SessionOptions &options) {
  return specfem::injection::ProviderSession(
      specfem::injection::PluginLibrary::open(mock_plugin_spec()), config,
      { 0.0, 1.0, nstep, {} }, options);
}

TEST(ProviderSessionMPI, RanksAgreeOnWindowSize) {
  reset_mock_calls();
  specfem::injection::SessionOptions options;
  // Rank 0 alone would choose 2 steps; ranks without points would choose all.
  options.memory_budget = 2 * (2 * 4 * 3 * sizeof(type_real));
  auto session = make_session("ok", options);
  session.setup(rank_points(), { { spfp_quantity_velocity }, {} });
  EXPECT_EQ(session.window_steps(), 2);

  session.ensure_window(5);
  EXPECT_EQ(session.frame().window_start, 4);
  EXPECT_EQ(mock_calls().fill, 1); // every rank fills, even without points
}

TEST(ProviderSessionMPI, FailureOnOneRankFailsAll) {
  reset_mock_calls();
  specfem::injection::SessionOptions options;
  options.window_steps = 5;
  auto session = make_session("fail-fill-rank0", options);
  session.setup(rank_points(), { { spfp_quantity_velocity }, {} });
  try {
    session.ensure_window(0);
    FAIL() << "every rank should throw";
  } catch (const std::runtime_error &error) {
    if (specfem::MPI::get_rank() == 0)
      EXPECT_THAT(error.what(), HasSubstr("mock fill failed"));
    else
      EXPECT_THAT(error.what(), HasSubstr("failed on another rank"));
  }
}

} // namespace specfem::injection_session_test
