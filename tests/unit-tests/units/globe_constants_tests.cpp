#include "specfem/globe/dimensionalization.hpp"
#include "specfem/globe/planet_constants.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {

using specfem::globe::Planet;
using specfem::globe::PlanetConstants;
using specfem::globe::PlanetConstantSet;

PlanetConstantSet earth_values() {
  return {
    .r_planet = 6371000.0,
    .rhoav = 5514.3,
    .one_minus_f_squared = (1.0 - 1.0 / 299.8) * (1.0 - 1.0 / 299.8),
    .hours_per_day = 24.0,
    .seconds_per_hour = 3600.0,
    .topo_maximum = 9000.0,
  };
}

TEST(GlobeConstants, EarthRotationConstants) {
  const PlanetConstants earth(Planet::earth, earth_values());
  EXPECT_DOUBLE_EQ(earth.values().hours_per_day, 24.0);
  const double two_omega =
      4.0 * std::acos(-1.0) /
      (earth.values().hours_per_day * earth.values().seconds_per_hour);
  EXPECT_NEAR(two_omega, 1.454441043328608e-4, 1.0e-18);
}

TEST(GlobeConstants, PreservesResolvedDatabaseValues) {
  auto values = earth_values();
  values.r_planet = 3389500.0;
  const PlanetConstants mars(Planet::mars, values);
  EXPECT_EQ(mars.planet(), Planet::mars);
  EXPECT_DOUBLE_EQ(mars.values().r_planet, 3389500.0);
}

TEST(GlobeConstants, LengthAndDensityRoundTrip) {
  const PlanetConstants earth(Planet::earth, earth_values());
  const specfem::units::Meters length(1234567.25);
  const auto length_nd = specfem::globe::nondimensionalize(length, earth);
  const auto recovered_length =
      specfem::globe::dimensionalize<specfem::units::Meters>(length_nd, earth);
  EXPECT_NEAR(recovered_length.raw(), length.raw(),
              8.0 * std::numeric_limits<type_real>::epsilon() * length.raw());

  const specfem::units::KilogramPerCubicMeter density(4876.5);
  const auto density_nd = specfem::globe::nondimensionalize(density, earth);
  const auto recovered_density =
      specfem::globe::dimensionalize<specfem::units::KilogramPerCubicMeter>(
          density_nd, earth);
  EXPECT_NEAR(recovered_density.raw(), density.raw(),
              8.0 * std::numeric_limits<type_real>::epsilon() * density.raw());
}

TEST(GlobeConstants, UnpopulatedRadiiAreReported) {
  const PlanetConstants earth(Planet::earth, earth_values());
  EXPECT_FALSE(earth.has_radii());
  EXPECT_THROW(static_cast<void>(earth.radii()), std::logic_error);
}

TEST(GlobeConstants, InconsistentRadiiAreRejected) {
  PlanetConstants earth(Planet::earth, earth_values());
  PlanetConstants::Radii radii{
    .r_icb = 1221500.0,
    .r_cmb = 3480000.0,
    .r_moho = 3400000.0,
    .r_80 = 6291000.0,
    .r_220 = 6151000.0,
    .r_400 = 5971000.0,
    .r_670 = 5701000.0,
    .r_771 = 5600000.0,
    .r_ocean = 6368000.0,
  };
  EXPECT_THROW(earth.set_radii(radii), std::runtime_error);
  EXPECT_FALSE(earth.has_radii());
}

TEST(GlobeConstants, ReplacingStoredRadiiWithMismatchIsRejected) {
  PlanetConstants earth(Planet::earth, earth_values());
  PlanetConstants::Radii radii{
    .r_icb = 1221500.0,
    .r_cmb = 3480000.0,
    .r_moho = 6346600.0,
    .r_80 = 6291000.0,
    .r_220 = 6151000.0,
    .r_400 = 5971000.0,
    .r_670 = 5701000.0,
    .r_771 = 5600000.0,
    .r_ocean = 6368000.0,
  };
  earth.set_radii(radii);
  radii.r_cmb += 1.0;
  EXPECT_THROW(earth.set_radii(radii), std::runtime_error);
}

} // namespace
