#include "specfem/constants/globe.hpp"
#include "specfem/utilities/dimensionalization.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {

using specfem::constants::Planet;
using specfem::constants::PlanetConstants;

TEST(GlobeConstants, EarthRotationConstants) {
  const PlanetConstants earth(Planet::earth);
  EXPECT_DOUBLE_EQ(earth.values().hours_per_day, 24.0);
  const double two_omega =
      4.0 * std::acos(-1.0) /
      (earth.values().hours_per_day * earth.values().seconds_per_hour);
  EXPECT_NEAR(two_omega, 1.454441043328608e-4, 1.0e-18);
}

TEST(GlobeConstants, PlanetTablesAreSelectableAndDistinct) {
  const PlanetConstants earth(Planet::earth);
  const PlanetConstants mars(Planet::mars);
  const PlanetConstants moon(Planet::moon);
  EXPECT_NE(mars.values().r_planet, earth.values().r_planet);
  EXPECT_NE(moon.values().r_planet, earth.values().r_planet);
  EXPECT_NE(mars.values().hours_per_day, earth.values().hours_per_day);
  EXPECT_NE(moon.values().rhoav, earth.values().rhoav);
}

TEST(GlobeConstants, LengthAndDensityRoundTrip) {
  const PlanetConstants earth(Planet::earth);
  const specfem::units::Meters length(1234567.25);
  const auto length_nd = specfem::utilities::nondimensionalize(length, earth);
  const auto recovered_length =
      specfem::utilities::dimensionalize<specfem::units::Meters>(length_nd,
                                                                 earth);
  EXPECT_NEAR(recovered_length.raw(), length.raw(),
              2.0 * std::numeric_limits<double>::epsilon() * length.raw());

  const specfem::units::KilogramPerCubicMeter density(4876.5);
  const auto density_nd = specfem::utilities::nondimensionalize(density, earth);
  const auto recovered_density =
      specfem::utilities::dimensionalize<specfem::units::KilogramPerCubicMeter>(
          density_nd, earth);
  EXPECT_NEAR(recovered_density.raw(), density.raw(),
              2.0 * std::numeric_limits<double>::epsilon() * density.raw());
}

TEST(GlobeConstants, DatabaseScaleMismatchThrows) {
  const PlanetConstants earth(Planet::earth);
  EXPECT_THROW(specfem::constants::check_database_values(earth, 3390000.0,
                                                         earth.values().rhoav),
               std::runtime_error);
  EXPECT_THROW(specfem::constants::check_database_values(
                   earth, earth.values().r_planet, 3393.0),
               std::runtime_error);
}

TEST(GlobeConstants, UnpopulatedRadiiAreReported) {
  const PlanetConstants earth(Planet::earth);
  EXPECT_FALSE(earth.has_radii());
  EXPECT_THROW(static_cast<void>(earth.radii()), std::logic_error);
}

TEST(GlobeConstants, InconsistentRadiiAreRejected) {
  PlanetConstants earth(Planet::earth);
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

} // namespace
