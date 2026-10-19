#include "specfem/constants.hpp"
#include "specfem/coordinate_systems/spherical_basis.hpp"
#include "specfem/source/force_rotation.hpp"

#include <cmath>
#include <gtest/gtest.h>

using specfem::sources::CartesianForce;
using specfem::sources::geographic_to_enu;
using specfem::sources::GeographicForce;
using specfem::sources::rotate_force_geographic_to_cartesian;

namespace {

constexpr double deg2rad = specfem::constants::degrees_to_radians_double;

std::pair<double, double> colat_lon(double lat_deg, double lon_deg) {
  return { M_PI / 2.0 - lat_deg * deg2rad, lon_deg * deg2rad };
}

GeographicForce mkf(double east, double north, double up) {
  return { static_cast<type_real>(east), static_cast<type_real>(north),
           static_cast<type_real>(up) };
}

double norm3(double a, double b, double c) {
  return std::sqrt(a * a + b * b + c * c);
}

constexpr double tol = 1e-5;

} // namespace

// At the North Pole (theta=0, phi=0): e_r = +z, e_theta = +x, e_phi = +y, so
// East -> +y, North (= -e_theta) -> -x, Up -> +z.
TEST(ForceRotation, NorthPoleAxes) {
  auto east = rotate_force_geographic_to_cartesian(0.0, 0.0, mkf(1, 0, 0));
  EXPECT_NEAR(east.fx, 0.0, tol);
  EXPECT_NEAR(east.fy, 1.0, tol);
  EXPECT_NEAR(east.fz, 0.0, tol);

  auto north = rotate_force_geographic_to_cartesian(0.0, 0.0, mkf(0, 1, 0));
  EXPECT_NEAR(north.fx, -1.0, tol);
  EXPECT_NEAR(north.fy, 0.0, tol);
  EXPECT_NEAR(north.fz, 0.0, tol);

  auto up = rotate_force_geographic_to_cartesian(0.0, 0.0, mkf(0, 0, 1));
  EXPECT_NEAR(up.fx, 0.0, tol);
  EXPECT_NEAR(up.fy, 0.0, tol);
  EXPECT_NEAR(up.fz, 1.0, tol);
}

// On the equator at longitude 0 (theta=pi/2, phi=0): e_r = +x, so Up -> +x,
// North (= -e_theta) -> +z, East (= e_phi) -> +y.
TEST(ForceRotation, EquatorLongitudeZeroAxes) {
  const auto [theta, phi] = colat_lon(0.0, 0.0);

  auto up = rotate_force_geographic_to_cartesian(theta, phi, mkf(0, 0, 1));
  EXPECT_NEAR(up.fx, 1.0, tol);
  EXPECT_NEAR(up.fy, 0.0, tol);
  EXPECT_NEAR(up.fz, 0.0, tol);

  auto north = rotate_force_geographic_to_cartesian(theta, phi, mkf(0, 1, 0));
  EXPECT_NEAR(north.fx, 0.0, tol);
  EXPECT_NEAR(north.fy, 0.0, tol);
  EXPECT_NEAR(north.fz, 1.0, tol);

  auto east = rotate_force_geographic_to_cartesian(theta, phi, mkf(1, 0, 0));
  EXPECT_NEAR(east.fx, 0.0, tol);
  EXPECT_NEAR(east.fy, 1.0, tol);
  EXPECT_NEAR(east.fz, 0.0, tol);
}

// A rotation preserves the vector's magnitude at every latitude and longitude.
TEST(ForceRotation, MagnitudePreserved) {
  const auto f = mkf(1.0, -2.0, 3.0);
  const double n0 = norm3(f.east, f.north, f.up);
  for (double lat = -80.0; lat <= 80.0; lat += 20.0) {
    for (double lon = 0.0; lon <= 360.0; lon += 45.0) {
      const auto [theta, phi] = colat_lon(lat, lon);
      const auto c = rotate_force_geographic_to_cartesian(theta, phi, f);
      SCOPED_TRACE("lat=" + std::to_string(lat) +
                   " lon=" + std::to_string(lon));
      EXPECT_NEAR(norm3(c.fx, c.fy, c.fz), n0, tol);
    }
  }
}

// The flat regional relabel is the identity for a vector (x=E, y=N, z=Up).
TEST(ForceRotation, EnuIsIdentity) {
  const auto c = geographic_to_enu(mkf(1, 2, 3));
  EXPECT_NEAR(c.fx, 1.0, tol);
  EXPECT_NEAR(c.fy, 2.0, tol);
  EXPECT_NEAR(c.fz, 3.0, tol);
}
