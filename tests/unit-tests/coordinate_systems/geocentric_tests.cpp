#include "specfem/coordinate_systems.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using specfem::coordinate_systems::cartesian_coordinates;
using specfem::coordinate_systems::geocentric_coordinates;
using specfem::coordinate_systems::geocentric_projection_config;
using specfem::coordinate_systems::geographic_coordinates;
using specfem::coordinate_systems::to;

namespace {

using cartesian3d =
    cartesian_coordinates<specfem::element::dimension_tag::dim3>;

constexpr double pi = std::numbers::pi;
constexpr double r_planet = 6371000.0; // m (arbitrary for the chain tests)

cartesian3d to_cartesian(const geocentric_coordinates &g) {
  return to<cartesian3d>(g);
}
geocentric_coordinates to_geocentric(const cartesian3d &c) {
  return to<geocentric_coordinates>(c);
}

// Smallest absolute angular difference in degrees, accounting for the 360 wrap
// (inverse returns longitude in [0, 360) in radians terms).
double longitude_error_deg(double a, double b) {
  double d = std::fmod(a - b, 360.0);
  if (d < -180.0)
    d += 360.0;
  if (d > 180.0)
    d -= 360.0;
  return std::abs(d);
}

} // namespace

// ── Known values (forward r,theta,phi -> x,y,z) ──────────────────────────────

TEST(CoordinateSystemsGeocentric, ForwardEquatorPrimeMeridian) {
  // theta = pi/2 (equator), phi = 0 -> +x axis.
  const auto c = to_cartesian({ r_planet, pi / 2.0, 0.0 });
  EXPECT_NEAR(c.x, r_planet, 1e-6);
  EXPECT_NEAR(c.y, 0.0, 1e-6);
  EXPECT_NEAR(c.z, 0.0, 1e-6);
}

TEST(CoordinateSystemsGeocentric, ForwardNorthPole) {
  // theta = 0 -> +z axis, independent of phi.
  const auto c = to_cartesian({ r_planet, 0.0, 1.234 });
  EXPECT_NEAR(c.x, 0.0, 1e-6);
  EXPECT_NEAR(c.y, 0.0, 1e-6);
  EXPECT_NEAR(c.z, r_planet, 1e-6);
}

TEST(CoordinateSystemsGeocentric, ForwardSouthPole) {
  const auto c = to_cartesian({ r_planet, pi, 1.234 });
  EXPECT_NEAR(c.x, 0.0, 1e-6);
  EXPECT_NEAR(c.y, 0.0, 1e-6);
  EXPECT_NEAR(c.z, -r_planet, 1e-6);
}

// ── Inverse at the poles must not NaN or hit a branch cut
// ─────────────────────

TEST(CoordinateSystemsGeocentric, InversePolesFinite) {
  const auto north = to_geocentric({ 0.0, 0.0, r_planet });
  EXPECT_NEAR(north.r, r_planet, 1e-6);
  // reduce() nudges points off the exact polar axis by ~1e-7 rad (matches
  // globe).
  EXPECT_NEAR(north.theta, 0.0, 1e-6);
  EXPECT_TRUE(std::isfinite(north.phi));

  const auto south = to_geocentric({ 0.0, 0.0, -r_planet });
  EXPECT_NEAR(south.r, r_planet, 1e-6);
  EXPECT_NEAR(south.theta, pi, 1e-9);
  EXPECT_TRUE(std::isfinite(south.phi));
}

TEST(CoordinateSystemsGeocentric, InverseLongitudeNormalizedToTwoPi) {
  // Point in the -y half plane: atan2 gives a negative angle; result must be
  // normalized into [0, 2*pi).
  const auto g = to_geocentric({ 1.0, -1.0, 0.0 });
  EXPECT_GE(g.phi, 0.0);
  EXPECT_LT(g.phi, 2.0 * pi);
  EXPECT_NEAR(g.phi, 7.0 * pi / 4.0, 1e-9); // -45 deg -> 315 deg
}

// ── Round trips (r,theta,phi) -> xyz -> (r,theta,phi) ────────────────────────

TEST(CoordinateSystemsGeocentric, RoundTripMidLatitude) {
  const geocentric_coordinates original{ r_planet - 12000.0, 0.9, 2.3 };
  const auto recovered = to_geocentric(to_cartesian(original));
  EXPECT_NEAR(recovered.r, original.r, 1e-6 * r_planet);
  EXPECT_NEAR(recovered.theta, original.theta, 1e-9);
  EXPECT_NEAR(recovered.phi, original.phi, 1e-9);
}

TEST(CoordinateSystemsGeocentric, RoundTripAntimeridian) {
  // phi = pi (antimeridian) must survive the round trip.
  const geocentric_coordinates original{ r_planet, pi / 3.0, pi };
  const auto recovered = to_geocentric(to_cartesian(original));
  EXPECT_NEAR(recovered.r, original.r, 1e-6 * r_planet);
  EXPECT_NEAR(recovered.theta, original.theta, 1e-9);
  EXPECT_NEAR(recovered.phi, original.phi, 1e-9);
}

// ── Full (lat, lon, depth) -> geocentric -> (lat, lon, depth) ────────────────
// Exercises the geographic <-> geocentric specializations directly (perfect
// sphere): colatitude from latitude (no flattening), r = r_planet - depth.

TEST(CoordinateSystemsGeocentric, GeographicChainRoundTrip) {
  const geocentric_projection_config config{ r_planet };

  struct Case {
    double lat, lon, depth;
  };
  const Case cases[] = {
    { 0.0, 0.0, 0.0 },        // equator, prime meridian
    { 90.0, 45.0, 0.0 },      // north pole (lon irrelevant)
    { -90.0, -120.0, 0.0 },   // south pole
    { 0.0, 180.0, 1000.0 },   // antimeridian, 1 km depth
    { 37.5, -122.3, 25000.0 } // mid-latitude, 25 km depth
  };

  for (const auto &k : cases) {
    const geographic_coordinates input(k.lon, k.lat, k.depth);

    const auto geocentric = to<geocentric_coordinates>(input, config);

    // Defining property of the spherical case.
    EXPECT_NEAR(geocentric.r, r_planet - k.depth, 1.0)
        << "radius != r_planet - depth for lat=" << k.lat;

    const auto back = to<geographic_coordinates>(geocentric, config);

    // Tolerance accommodates globe reduce()'s ~1e-7 rad axis nudge (~6e-6 deg)
    // at points on the poles / prime meridian.
    EXPECT_NEAR(back.latitude, k.lat, 1e-4) << "latitude round trip";
    EXPECT_NEAR(back.depth, k.depth, 1.0) << "depth round trip";
    if (std::abs(k.lat) < 90.0) // longitude is undefined at the poles
      EXPECT_NEAR(longitude_error_deg(back.longitude, k.lon), 0.0, 1e-4)
          << "longitude round trip";
  }
}
