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
#include <vector>

namespace specfem::injection_session_test {

using ::testing::HasSubstr;

constexpr int nstep = 10;

// Rank 0 holds four elastic points; every other rank holds the same group
// with no points.
specfem::injection::InjectionPoints rank_points(
    const specfem_property property = specfem_property_isotropic,
    const std::vector<specfem::injection::QuantityRequest> &quantities = {
        { specfem_quantity_velocity, 3 } }) {
  const int nrows = specfem::MPI::get_rank() == 0 ? 4 : 0;
  specfem::injection::InjectionPoints points(nrows);
  for (int row = 0; row < nrows; ++row)
    points.x(row) = static_cast<type_real>(row);
  specfem::injection::PointGroup group;
  group.medium = specfem_medium_elastic;
  group.property = property;
  group.count = nrows;
  group.properties =
      specfem::injection::PointGroup::properties_view("properties", nrows, 0);
  group.quantities = quantities;
  points.groups.push_back(group);
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
  session.setup(rank_points());
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
  session.setup(rank_points());
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

TEST(ProviderSessionMPI, RanksMustHoldSameGroups) {
  reset_mock_calls();
  specfem::injection::SessionOptions options;
  options.window_steps = 5;
  auto session = make_session("ok", options);
  const auto property = specfem::MPI::get_rank() == 0
                            ? specfem_property_isotropic
                            : specfem_property_anisotropic;
  try {
    session.setup(rank_points(property));
    FAIL() << "every rank should throw";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(),
                HasSubstr("every rank must hold the same groups"));
  }
  EXPECT_EQ(mock_calls().setup, 0);
}

class ProviderSessionMPIQuantities
    : public ::testing::TestWithParam<
          std::vector<specfem::injection::QuantityRequest>> {};

// Rank 0 requests velocity with 3 components and the other ranks request the
// parameter; the mismatch must fail setup on every rank before the plugin is
// set up.
TEST_P(ProviderSessionMPIQuantities, RanksMustRequestSameQuantities) {
  reset_mock_calls();
  specfem::injection::SessionOptions options;
  options.window_steps = 5;
  auto session = make_session("ok", options);
  const auto quantities =
      specfem::MPI::get_rank() == 0
          ? std::vector<specfem::injection::
                            QuantityRequest>{ { specfem_quantity_velocity, 3 } }
          : GetParam();
  try {
    session.setup(rank_points(specfem_property_isotropic, quantities));
    FAIL() << "every rank should throw";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(),
                HasSubstr("every rank must hold the same groups"));
  }
  EXPECT_EQ(mock_calls().setup, 0);
}

INSTANTIATE_TEST_SUITE_P(
    Mismatches, ProviderSessionMPIQuantities,
    ::testing::Values(std::vector<specfem::injection::QuantityRequest>{},
                      std::vector<specfem::injection::QuantityRequest>{
                          { specfem_quantity_velocity, 3 },
                          { specfem_quantity_traction, 3 } },
                      std::vector<specfem::injection::QuantityRequest>{
                          { specfem_quantity_displacement, 3 } },
                      std::vector<specfem::injection::QuantityRequest>{
                          { specfem_quantity_velocity, 2 } }));

} // namespace specfem::injection_session_test
