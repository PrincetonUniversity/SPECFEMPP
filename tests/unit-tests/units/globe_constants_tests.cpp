#include "specfem/globe/planet_constants.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace globe_constants_test_impl {

using specfem::globe::Planet;
using specfem::globe::PlanetConstants;

constexpr int earth_schema_version =
    PlanetConstants::current_schema_version(Planet::earth);

std::vector<double> earth_values() {
  return { 6371000.0, 5514.3, (1.0 - 1.0 / 299.8) * (1.0 - 1.0 / 299.8),
           24.0,      3600.0, 9000.0 };
}

TEST(GlobeConstants, PreservesDatabasePlanetConstants) {
  const PlanetConstants constants = PlanetConstants::from_database(
      Planet::earth, earth_schema_version,
      { 7000000.0, 5000.0, 0.99, 25.0, 3601.0, 10000.0 });

  EXPECT_EQ(constants.planet(), Planet::earth);
  EXPECT_DOUBLE_EQ(constants.r_planet(), 7000000.0);
  EXPECT_DOUBLE_EQ(constants.rhoav(), 5000.0);
  EXPECT_DOUBLE_EQ(constants.one_minus_f_squared(), 0.99);
  EXPECT_DOUBLE_EQ(constants.hours_per_day(), 25.0);
  EXPECT_DOUBLE_EQ(constants.seconds_per_hour(), 3601.0);
  EXPECT_DOUBLE_EQ(constants.topo_maximum(), 10000.0);
}

TEST(GlobeConstants, AcceptsResolvedMarsRadiusOverride) {
  const PlanetConstants mars = PlanetConstants::from_database(
      Planet::mars, PlanetConstants::current_schema_version(Planet::mars),
      { 3389500.0, 3393.0, (1.0 - 1.0 / 169.8) * (1.0 - 1.0 / 169.8), 24.658,
        3600.0, 23200.0 });
  EXPECT_DOUBLE_EQ(mars.r_planet(), 3389500.0);
}

TEST(GlobeConstants, RejectsUnknownPlanetSchema) {
  EXPECT_THROW(static_cast<void>(PlanetConstants::from_database(
                   Planet::earth, 1, earth_values())),
               std::runtime_error);
}

TEST(GlobeConstants, RejectsIncorrectPlanetValueCount) {
  auto values = earth_values();
  values.pop_back();
  EXPECT_THROW(static_cast<void>(PlanetConstants::from_database(
                   Planet::earth, earth_schema_version, std::move(values))),
               std::runtime_error);
}

TEST(GlobeConstants, RejectsInvalidPlanetValues) {
  auto nonfinite = earth_values();
  nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(static_cast<void>(PlanetConstants::from_database(
                   Planet::earth, earth_schema_version, std::move(nonfinite))),
               std::runtime_error);

  auto nonpositive = earth_values();
  nonpositive[4] = 0.0;
  EXPECT_THROW(
      static_cast<void>(PlanetConstants::from_database(
          Planet::earth, earth_schema_version, std::move(nonpositive))),
      std::runtime_error);

  auto invalid_flattening = earth_values();
  invalid_flattening[2] = 1.01;
  EXPECT_THROW(
      static_cast<void>(PlanetConstants::from_database(
          Planet::earth, earth_schema_version, std::move(invalid_flattening))),
      std::runtime_error);
}

} // namespace globe_constants_test_impl
