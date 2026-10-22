#include "specfem/coordinate_systems.hpp"
#include "specfem/coordinate_systems/spherical_basis.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

namespace spherical_basis_test_impl {

using Vec3 = std::array<double, 3>;

double dot(const Vec3 &a, const Vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0] };
}

double determinant(const specfem::coordinate_systems::GeographicBasis &basis) {
  return dot(basis.east, cross(basis.north, basis.up));
}

void expect_vec_near(const Vec3 &actual, const Vec3 &expected,
                     double tolerance) {
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "component " << i;
  }
}

void expect_orthonormal_right_handed(
    const specfem::coordinate_systems::GeographicBasis &basis,
    double tolerance) {
  EXPECT_NEAR(dot(basis.east, basis.east), 1.0, tolerance);
  EXPECT_NEAR(dot(basis.north, basis.north), 1.0, tolerance);
  EXPECT_NEAR(dot(basis.up, basis.up), 1.0, tolerance);
  EXPECT_NEAR(dot(basis.east, basis.north), 0.0, tolerance);
  EXPECT_NEAR(dot(basis.east, basis.up), 0.0, tolerance);
  EXPECT_NEAR(dot(basis.north, basis.up), 0.0, tolerance);
  expect_vec_near(cross(basis.east, basis.north), basis.up, tolerance);
  EXPECT_NEAR(determinant(basis), 1.0, tolerance);
}

constexpr double planet_radius = 6371000.0;

} // namespace spherical_basis_test_impl

TEST(GeographicBasisTest, OrthonormalAndRightHanded) {
  for (int latitude = -80; latitude <= 80; latitude += 20) {
    for (int longitude = 0; longitude < 360; longitude += 45) {
      const double theta =
          std::numbers::pi / 2.0 - latitude * std::numbers::pi / 180.0;
      const double phi = longitude * std::numbers::pi / 180.0;
      const auto basis =
          specfem::coordinate_systems::local_geographic_basis(theta, phi);
      spherical_basis_test_impl::expect_orthonormal_right_handed(basis, 1e-12);
    }
  }
}

TEST(GeographicBasisTest, EquatorLongitudeZeroAxes) {
  const auto basis = specfem::coordinate_systems::local_geographic_basis(
      std::numbers::pi / 2.0, 0.0);
  spherical_basis_test_impl::expect_vec_near(basis.east, { 0.0, 1.0, 0.0 },
                                             1e-12);
  spherical_basis_test_impl::expect_vec_near(basis.north, { 0.0, 0.0, 1.0 },
                                             1e-12);
  spherical_basis_test_impl::expect_vec_near(basis.up, { 1.0, 0.0, 0.0 },
                                             1e-12);
}

TEST(GeographicBasisTest, NorthPoleAxes) {
  const auto basis =
      specfem::coordinate_systems::local_geographic_basis(0.0, 0.0);
  spherical_basis_test_impl::expect_vec_near(basis.east, { 0.0, 1.0, 0.0 },
                                             1e-12);
  spherical_basis_test_impl::expect_vec_near(basis.north, { -1.0, 0.0, 0.0 },
                                             1e-12);
  spherical_basis_test_impl::expect_vec_near(basis.up, { 0.0, 0.0, 1.0 },
                                             1e-12);
}

TEST(GeographicBasisTest, PoleThroughCartesianInverseIsFinite) {
  using cartesian3d = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;

  struct PoleCase {
    double z;
    spherical_basis_test_impl::Vec3 east;
    spherical_basis_test_impl::Vec3 north;
    spherical_basis_test_impl::Vec3 up;
  };
  const PoleCase cases[] = {
    { spherical_basis_test_impl::planet_radius,
      { 0.0, 1.0, 0.0 },
      { -1.0, 0.0, 0.0 },
      { 0.0, 0.0, 1.0 } },
    { -spherical_basis_test_impl::planet_radius,
      { 0.0, 1.0, 0.0 },
      { 1.0, 0.0, 0.0 },
      { 0.0, 0.0, -1.0 } },
  };

  for (const auto &pole : cases) {
    const auto geocentric = specfem::coordinate_systems::to<
        specfem::coordinate_systems::geocentric_coordinates>(
        cartesian3d(0.0, 0.0, pole.z));
    const auto basis = specfem::coordinate_systems::local_geographic_basis(
        geocentric.theta, geocentric.phi);

    for (const auto *vector : { &basis.east, &basis.north, &basis.up }) {
      for (const double component : *vector) {
        EXPECT_TRUE(std::isfinite(component));
      }
    }
    spherical_basis_test_impl::expect_vec_near(basis.east, pole.east, 1e-6);
    spherical_basis_test_impl::expect_vec_near(basis.north, pole.north, 1e-6);
    spherical_basis_test_impl::expect_vec_near(basis.up, pole.up, 1e-6);
    spherical_basis_test_impl::expect_orthonormal_right_handed(basis, 1e-6);
  }
}

TEST(GeographicBasisTest, RoundTrip) {
  const spherical_basis_test_impl::Vec3 vector = { 0.3, -1.2, 2.5 };
  const double angles[][2] = {
    { 0.4, 0.7 }, { std::numbers::pi / 2.0, 0.0 }, { 2.1, 4.5 }, { 2.9, 6.0 }
  };
  for (const auto &angle : angles) {
    const auto basis =
        specfem::coordinate_systems::local_geographic_basis(angle[0], angle[1]);
    const double e = spherical_basis_test_impl::dot(vector, basis.east);
    const double n = spherical_basis_test_impl::dot(vector, basis.north);
    const double u = spherical_basis_test_impl::dot(vector, basis.up);
    spherical_basis_test_impl::Vec3 reconstructed;
    for (int i = 0; i < 3; ++i) {
      reconstructed[i] =
          e * basis.east[i] + n * basis.north[i] + u * basis.up[i];
    }
    spherical_basis_test_impl::expect_vec_near(reconstructed, vector, 1e-12);
  }
}

TEST(GeographicBasisTest, MatchesSphericalBasis) {
  const double angles[][2] = { { 0.4, 0.7 }, { 1.3, 2.2 }, { 2.6, 5.1 } };
  for (const auto &angle : angles) {
    const auto spherical =
        specfem::coordinate_systems::local_spherical_basis(angle[0], angle[1]);
    const auto geographic =
        specfem::coordinate_systems::local_geographic_basis(angle[0], angle[1]);
    spherical_basis_test_impl::expect_vec_near(geographic.east, spherical.e_phi,
                                               1e-15);
    spherical_basis_test_impl::expect_vec_near(
        geographic.north,
        { -spherical.e_theta[0], -spherical.e_theta[1], -spherical.e_theta[2] },
        1e-15);
    spherical_basis_test_impl::expect_vec_near(geographic.up, spherical.e_r,
                                               1e-15);
  }
}
