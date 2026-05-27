#include "specfem/coordinate_systems.hpp"
#include <cmath>
#include <gtest/gtest.h>

constexpr auto dim3 = specfem::element::dimension_tag::dim3;

using specfem::coordinate_systems::cartesian_coordinates;
using specfem::coordinate_systems::geographic_coordinates;
using specfem::coordinate_systems::transform;
using specfem::coordinate_systems::utm_projection_config;

// Reference values from the corrected Fortran utm_geo.f90 comments:
//   477415.5  5712313.5  <->  lon 2.6741959317615298  lat 51.561449479910003
//   (zone 31, northern hemisphere)
// Round-trip error should be < 0.001 m.

TEST(CoordinateSystemsUtm, ForwardConversionZone31) {
  const geographic_coordinates geo{ 2.6741959317615298, 51.561449479910003,
                                    0.0 };
  const auto cart =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ 31 });

  EXPECT_NEAR(cart.x, 477415.5, 0.01)
      << "Easting mismatch for zone 31 forward conversion";
  EXPECT_NEAR(cart.y, 5712313.5, 0.01)
      << "Northing mismatch for zone 31 forward conversion";
}

TEST(CoordinateSystemsUtm, InverseConversionZone31) {
  const cartesian_coordinates<dim3> cart{ 477415.5, 5712313.5, 0.0 };
  const auto geo =
      transform<geographic_coordinates>(cart, utm_projection_config{ 31 });

  EXPECT_NEAR(geo.longitude, 2.6741959317615298, 1e-9)
      << "Longitude mismatch for zone 31 inverse conversion";
  EXPECT_NEAR(geo.latitude, 51.561449479910003, 1e-9)
      << "Latitude mismatch for zone 31 inverse conversion";
}

TEST(CoordinateSystemsUtm, RoundTripForwardInverse) {
  const geographic_coordinates original{ -73.9857, 40.7484, 0.0 };
  const auto cart = transform<cartesian_coordinates<dim3>>(
      original, utm_projection_config{ 18 });
  const auto recovered =
      transform<geographic_coordinates>(cart, utm_projection_config{ 18 });

  EXPECT_NEAR(recovered.longitude, original.longitude, 1e-8)
      << "Round-trip longitude error exceeds tolerance";
  EXPECT_NEAR(recovered.latitude, original.latitude, 1e-8)
      << "Round-trip latitude error exceeds tolerance";
}

TEST(CoordinateSystemsUtm, RoundTripInverseForward) {
  const cartesian_coordinates<dim3> original{ 500000.0, 4500000.0, 0.0 };
  const auto geo =
      transform<geographic_coordinates>(original, utm_projection_config{ 15 });
  const auto recovered =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ 15 });

  EXPECT_NEAR(recovered.x, original.x, 0.001)
      << "Round-trip easting error exceeds 1 mm";
  EXPECT_NEAR(recovered.y, original.y, 0.001)
      << "Round-trip northing error exceeds 1 mm";
}

TEST(CoordinateSystemsUtm, NorthPole) {
  const geographic_coordinates geo{ 0.0, 90.0, 0.0 };
  const auto cart =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ 31 });

  EXPECT_NEAR(cart.x, 500000.0, 0.01)
      << "North pole easting should be false easting";
  EXPECT_GT(cart.y, 0.0) << "North pole northing should be positive";
}

TEST(CoordinateSystemsUtm, SouthPole) {
  const geographic_coordinates geo{ 0.0, -90.0, 0.0 };
  const auto cart =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ 31 });

  EXPECT_NEAR(cart.x, 500000.0, 0.01)
      << "South pole easting should be false easting";
}

TEST(CoordinateSystemsUtm, SouthernHemisphere) {
  const geographic_coordinates geo{ 151.2093, -33.8688, 0.0 };
  const auto cart =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ -56 });

  EXPECT_GT(cart.y, 0.0)
      << "Southern hemisphere northing should be positive (with 10M offset)";
  EXPECT_LT(cart.y, 1.0e7)
      << "Southern hemisphere northing should be < 10,000 km";

  // Round-trip
  const auto recovered =
      transform<geographic_coordinates>(cart, utm_projection_config{ -56 });
  EXPECT_NEAR(recovered.longitude, geo.longitude, 1e-8);
  EXPECT_NEAR(recovered.latitude, geo.latitude, 1e-8);
}

TEST(CoordinateSystemsUtm, SuppressProjectionForward) {
  const geographic_coordinates geo{ 12.34, 56.78, 100.0 };
  const auto cart = transform<cartesian_coordinates<dim3>>(
      geo, utm_projection_config{ 31, true });

  EXPECT_DOUBLE_EQ(cart.x, geo.longitude)
      << "Suppress projection should pass longitude through as x";
  EXPECT_DOUBLE_EQ(cart.y, geo.latitude)
      << "Suppress projection should pass latitude through as y";
  EXPECT_DOUBLE_EQ(cart.z, geo.depth)
      << "Suppress projection should pass depth through as z";
  // Suppress mode sets origin to nullopt (needs resolution)
  EXPECT_FALSE(cart.origin.has_value())
      << "Suppress projection should leave origin unset";
}

TEST(CoordinateSystemsUtm, SuppressProjectionInverse) {
  const cartesian_coordinates<dim3> cart{ 12.34, 56.78, 200.0 };
  const auto geo = transform<geographic_coordinates>(
      cart, utm_projection_config{ 31, true });

  EXPECT_DOUBLE_EQ(geo.longitude, cart.x)
      << "Suppress projection should pass x through as longitude";
  EXPECT_DOUBLE_EQ(geo.latitude, cart.y)
      << "Suppress projection should pass y through as latitude";
  EXPECT_DOUBLE_EQ(geo.depth, cart.z)
      << "Suppress projection should pass z through as depth";
}

TEST(CoordinateSystemsUtm, LongitudeWrapping) {
  const geographic_coordinates geo{ 179.5, 45.0, 0.0 };
  const auto cart =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ 60 });
  const auto recovered =
      transform<geographic_coordinates>(cart, utm_projection_config{ 60 });

  EXPECT_NEAR(recovered.longitude, geo.longitude, 1e-8)
      << "Longitude wrapping near 180 degrees";
  EXPECT_NEAR(recovered.latitude, geo.latitude, 1e-8);
}

TEST(CoordinateSystemsUtm, NegativeLongitudeWrapping) {
  const geographic_coordinates geo{ -179.5, 45.0, 0.0 };
  const auto cart =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ 1 });
  const auto recovered =
      transform<geographic_coordinates>(cart, utm_projection_config{ 1 });

  EXPECT_NEAR(recovered.longitude, geo.longitude, 1e-8)
      << "Longitude wrapping near -180 degrees";
  EXPECT_NEAR(recovered.latitude, geo.latitude, 1e-8);
}

TEST(CoordinateSystemsUtm, Equator) {
  const geographic_coordinates geo{ 3.0, 0.0, 0.0 };
  const auto cart =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ 31 });

  EXPECT_NEAR(cart.x, 500000.0, 0.01)
      << "Easting at central meridian should be ~500000";
  EXPECT_NEAR(cart.y, 0.0, 0.01) << "Northing at equator should be ~0";
}

TEST(CoordinateSystemsUtm, MultipleZonesRoundTrip) {
  const struct {
    double lon;
    double lat;
    int zone;
  } test_cases[] = {
    { -122.4194, 37.7749, 10 },  // San Francisco
    { 139.6917, 35.6895, 54 },   // Tokyo
    { 2.3522, 48.8566, 31 },     // Paris
    { -43.1729, -22.9068, -23 }, // Rio de Janeiro (southern hemisphere)
    { 18.0686, -33.9249, -34 },  // Cape Town (southern hemisphere)
  };

  for (const auto &tc : test_cases) {
    const geographic_coordinates geo{ tc.lon, tc.lat, 0.0 };
    const auto cart = transform<cartesian_coordinates<dim3>>(
        geo, utm_projection_config{ tc.zone });
    const auto recovered = transform<geographic_coordinates>(
        cart, utm_projection_config{ tc.zone });

    EXPECT_NEAR(recovered.longitude, tc.lon, 1e-8)
        << "Round-trip longitude failed for zone " << tc.zone;
    EXPECT_NEAR(recovered.latitude, tc.lat, 1e-8)
        << "Round-trip latitude failed for zone " << tc.zone;
  }
}

TEST(CoordinateSystemsUtm, DepthNegatedInForward) {
  const geographic_coordinates geo{ 2.6741959317615298, 51.561449479910003,
                                    1234.5 };
  const auto cart =
      transform<cartesian_coordinates<dim3>>(geo, utm_projection_config{ 31 });

  // Forward transform negates depth to z: z = -depth
  EXPECT_DOUBLE_EQ(cart.z, -1234.5)
      << "Depth should be negated to z in forward transform";
  // Origin is nullopt (needs topographic resolution)
  EXPECT_FALSE(cart.origin.has_value())
      << "Forward transform should leave origin unset for depth resolution";
}

TEST(CoordinateSystemsUtm, DepthPassThroughInverse) {
  const cartesian_coordinates<dim3> cart{ 477415.5, 5712313.5, -5678.9 };
  const auto geo =
      transform<geographic_coordinates>(cart, utm_projection_config{ 31 });

  // Inverse passes z through as depth (caller is responsible for sign)
  EXPECT_DOUBLE_EQ(geo.depth, -5678.9)
      << "z should be passed through unchanged to depth";
}
