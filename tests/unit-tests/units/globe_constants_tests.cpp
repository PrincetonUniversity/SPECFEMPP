#include "specfem/globe/planet_constants.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <utility>
#include <vector>

namespace globe_constants_test_impl {

using specfem::globe::Planet;
using specfem::globe::PlanetConstants;

constexpr int earth_schema_version =
    PlanetConstants::current_schema_version(Planet::earth);

std::vector<double> earth_values() {
  return {
    6371000.0, 5514.3,    (1.0 - 1.0 / 299.8) * (1.0 - 1.0 / 299.8),
    24.0,      3600.0,    9000.0,
    1221500.0, 3480000.0, 6346600.0,
    6291000.0, 6151000.0, 5971000.0,
    5701000.0, 5600000.0, 6368000.0,
  };
}

TEST(GlobeConstants, PreservesSelectedPlanetSchema) {
  const PlanetConstants earth = PlanetConstants::from_database(
      Planet::earth, earth_schema_version, earth_values());

  EXPECT_EQ(earth.planet(), Planet::earth);
  EXPECT_EQ(earth.schema_version(), earth_schema_version);
  EXPECT_EQ(earth.values(), earth_values());
}

TEST(GlobeConstants, RejectsUnknownPlanetSchema) {
  EXPECT_THROW(static_cast<void>(PlanetConstants::from_database(
                   Planet::earth, 2, earth_values())),
               std::runtime_error);
}

TEST(GlobeConstants, RejectsIncorrectPlanetValueCount) {
  auto values = earth_values();
  values.pop_back();
  EXPECT_THROW(static_cast<void>(PlanetConstants::from_database(
                   Planet::earth, earth_schema_version, std::move(values))),
               std::runtime_error);
}

TEST(GlobeConstants, RejectsNonFinitePlanetValues) {
  auto values = earth_values();
  values[8] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(static_cast<void>(PlanetConstants::from_database(
                   Planet::earth, earth_schema_version, std::move(values))),
               std::runtime_error);
}

TEST(GlobeConstants, RejectsValuesThatDisagreeWithCatalog) {
  const PlanetConstants earth = PlanetConstants::from_database(
      Planet::earth, earth_schema_version, earth_values());
  auto catalog_values = earth_values();
  catalog_values[7] += 1.0;

  EXPECT_THROW(earth.check_catalog_values(catalog_values), std::runtime_error);
}

} // namespace globe_constants_test_impl
