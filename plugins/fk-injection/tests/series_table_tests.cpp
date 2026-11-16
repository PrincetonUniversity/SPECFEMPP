#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "series_table.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

using fk_injection::type_real;

namespace series_table_tests_impl {

/** @brief Elastic-only 3-layer model: two finite layers + half-space. */
fk_injection::LayeredModel make_elastic_model() {
  fk_injection::ElasticIsotropicLayer layer0;
  layer0.density = 2800;
  layer0.p_velocity = 6000;
  layer0.s_velocity = 3500;
  layer0.thickness = 10000;

  fk_injection::ElasticIsotropicLayer layer1;
  layer1.density = 2900;
  layer1.p_velocity = 6800;
  layer1.s_velocity = 3900;
  layer1.thickness = 20000;

  fk_injection::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300;
  halfspace.p_velocity = 8100;
  halfspace.s_velocity = 4500;
  halfspace.thickness = 0;

  return fk_injection::LayeredModel({}, { layer0, layer1, halfspace });
}

/**
 * @brief Acoustic + elastic model: two fluid layers on top, crust + half-space.
 *
 * Layer structure (depths positive-downward):
 *   0–2000  m  : fluid vp=2000
 *   2000–3000 m: fluid vp=1000
 *   3000–13000m: elastic crust
 *   13000+     : elastic half-space
 */
fk_injection::LayeredModel make_a1_model() {
  fk_injection::AcousticLayer fluid0;
  fluid0.density = 1000;
  fluid0.p_velocity = 2000;
  fluid0.thickness = 2000;

  fk_injection::AcousticLayer fluid1;
  fluid1.density = 900;
  fluid1.p_velocity = 1000;
  fluid1.thickness = 1000;

  fk_injection::ElasticIsotropicLayer crust;
  crust.density = 2800;
  crust.p_velocity = 6000;
  crust.s_velocity = 3500;
  crust.thickness = 10000;

  fk_injection::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300;
  halfspace.p_velocity = 8100;
  halfspace.s_velocity = 4500;
  halfspace.thickness = 0;

  return fk_injection::LayeredModel({ fluid0, fluid1 }, { crust, halfspace });
}

/** @brief Response to a P wave at 20 deg take-off, phi = 0.3 rad. */
fk_injection::PlaneWaveResponse
make_response(const fk_injection::LayeredModel &model) {
  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.azimuth_phi = static_cast<type_real>(0.3);
  wave.take_off_theta = static_cast<type_real>(20.0 * 3.14159265358979 / 180.0);
  wave.origin_z = static_cast<type_real>(-model.halfspace_top_depth());
  wave.gaussian_half_duration = 2;

  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 200;
  window.frequency_sampling = 4;
  window.time_window_length = 20;
  return fk_injection::PlaneWaveResponse(model, wave, window, {});
}

/** @brief Elastic keys at the depths. */
std::vector<fk_injection::SeriesKey>
elastic_keys(const std::vector<double> &depths) {
  std::vector<fk_injection::SeriesKey> keys;
  for (const double depth : depths)
    keys.push_back(
        { specfem::plugin::Medium::elastic, static_cast<type_real>(depth) });
  return keys;
}

/** @brief Number of rows for elastic keys at the depths. */
int number_of_rows(const std::vector<double> &depths) {
  const auto model = make_elastic_model();
  return fk_injection::SeriesTable::synthesize(make_response(model),
                                               elastic_keys(depths))
      .number_of_rows();
}

} // namespace series_table_tests_impl

TEST(SeriesTable, DistinctDepthsFormDistinctRows) {
  const auto model = series_table_tests_impl::make_elastic_model();
  // Surface, layer 0, layer 1, half-space.
  const auto keys =
      series_table_tests_impl::elastic_keys({ 0.0, 5000.0, 15000.0, 35000.0 });
  const auto table = fk_injection::SeriesTable::synthesize(
      series_table_tests_impl::make_response(model), keys);

  ASSERT_EQ(table.number_of_rows(), 4);
  std::set<int> rows;
  for (const auto &key : keys) {
    const int row = table.row(key);
    rows.insert(row);
    EXPECT_EQ(table.location(row).depth, static_cast<double>(key.depth));
  }
  EXPECT_EQ(rows.size(), 4u);
}

// Many points at few depths, as on a structured mesh's faces: one row per
// depth, each located at that depth.
TEST(SeriesTable, RowsFollowDistinctDepthsNotPoints) {
  const auto model = series_table_tests_impl::make_elastic_model();
  const std::vector<double> depths = { 4000.0, 12000.0, 20000.0, 40000.0 };
  std::vector<double> repeated;
  for (int copy = 0; copy < 25; ++copy)
    for (const double depth : depths)
      repeated.push_back(depth);
  const auto keys = series_table_tests_impl::elastic_keys(repeated);
  const auto table = fk_injection::SeriesTable::synthesize(
      series_table_tests_impl::make_response(model), keys);

  EXPECT_EQ(table.number_of_rows(), static_cast<int>(depths.size()));
  for (const auto &key : keys)
    EXPECT_EQ(table.location(table.row(key)).depth,
              static_cast<double>(key.depth));
}

TEST(SeriesTable, IdenticalDepthsMergeDistinctSeparate) {
  EXPECT_EQ(series_table_tests_impl::number_of_rows({ 5000.0, 5000.0 }), 1)
      << "identical depths must share one row";
  EXPECT_EQ(series_table_tests_impl::number_of_rows({ 5000.0, 5000.5 }), 2)
      << "half-metre separation must form distinct rows";
}

// Depths one float ulp apart are the same mesh depth up to round-off.
TEST(SeriesTable, RoundOffMerges) {
  const auto model = series_table_tests_impl::make_elastic_model();
  const float depth = 5000.0f;
  const float next = std::nextafterf(depth, 1.0e9f);
  const auto keys = series_table_tests_impl::elastic_keys(
      { static_cast<double>(next), static_cast<double>(depth) });
  const auto table = fk_injection::SeriesTable::synthesize(
      series_table_tests_impl::make_response(model), keys);

  ASSERT_EQ(table.number_of_rows(), 1);
  EXPECT_EQ(table.location(0).depth, static_cast<double>(depth))
      << "the row sits at the shallowest depth";
  EXPECT_EQ(table.row(keys[0]), 0);
  EXPECT_EQ(table.row(keys[1]), 0);
}

TEST(SeriesTable, DistinctDepthLevelsDoNotMerge) {
  std::vector<double> spaced;
  for (int k = 0; k < 6; ++k)
    spaced.push_back(5000.0 + k * 1000.0);
  EXPECT_EQ(series_table_tests_impl::number_of_rows(spaced), 6)
      << "six depths 1000 m apart must form six rows";

  EXPECT_EQ(series_table_tests_impl::number_of_rows(
                { 5000.0, 5000.0, 5000.0, 8000.0, 8000.0, 8000.0 }),
            2)
      << "two repeated depths must collapse to two rows";
}

// Both media at the fluid-solid interface: two rows, elastic first.
TEST(SeriesTable, MediumSplitsRows) {
  const auto model = series_table_tests_impl::make_a1_model();
  const std::vector<fk_injection::SeriesKey> keys = {
    { specfem::plugin::Medium::acoustic, 3000 },
    { specfem::plugin::Medium::elastic, 3000 }
  };
  const auto table = fk_injection::SeriesTable::synthesize(
      series_table_tests_impl::make_response(model), keys);

  ASSERT_EQ(table.number_of_rows(), 2);
  EXPECT_EQ(table.row(keys[1]), 0);
  EXPECT_EQ(table.row(keys[0]), 1);
  EXPECT_EQ(table.location(0).medium, specfem::plugin::Medium::elastic);
  EXPECT_EQ(table.location(1).medium, specfem::plugin::Medium::acoustic);
}

// Round-off slack only: 8 float ulps of the deepest key.
TEST(SeriesTable, ToleranceIsFloatRoundOff) {
  const auto model = series_table_tests_impl::make_elastic_model();
  const auto table = fk_injection::SeriesTable::synthesize(
      series_table_tests_impl::make_response(model),
      series_table_tests_impl::elastic_keys({ 1000.0, 20000.0 }));

  const float depth = 20000.0f;
  const float next =
      std::nextafterf(depth, std::numeric_limits<float>::infinity());
  EXPECT_EQ(table.tolerance(), 8.0 * static_cast<double>(next - depth));
}

TEST(SeriesTable, RowRejectsAKeyItWasNotBuiltFor) {
  const auto model = series_table_tests_impl::make_a1_model();
  const auto table = fk_injection::SeriesTable::synthesize(
      series_table_tests_impl::make_response(model),
      { { specfem::plugin::Medium::elastic, 5000 } });

  EXPECT_THROW(table.row({ specfem::plugin::Medium::elastic, 6000 }),
               std::runtime_error);
  EXPECT_THROW(table.row({ specfem::plugin::Medium::elastic, 4000 }),
               std::runtime_error);
  EXPECT_THROW(table.row({ specfem::plugin::Medium::acoustic, 1000 }),
               std::runtime_error);
}

TEST(SeriesTable, SynthesizeRejectsAKeyTheModelCannotServe) {
  const auto model = series_table_tests_impl::make_a1_model();
  EXPECT_THROW(fk_injection::SeriesTable::synthesize(
                   series_table_tests_impl::make_response(model),
                   { { specfem::plugin::Medium::elastic, 1000 } }),
               std::runtime_error);
}
