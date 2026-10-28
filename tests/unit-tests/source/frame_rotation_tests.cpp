#include "specfem/constants.hpp"
#include "specfem/coordinate_systems/basis_rotation.hpp"
#include "specfem/coordinate_systems/east_north_up.hpp"
#include "specfem/source/force_vector.hpp"
#include "specfem/source/moment_tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <string>
#include <utility>

namespace specfem::sources::frame_rotation_test_impl {

using Cartesian = specfem::coordinate_systems::CartesianFrame;
using Geocentric = specfem::coordinate_systems::GeocentricFrame;
using CartesianTensor = specfem::sources::MomentTensor<Cartesian>;
using GeocentricTensor = specfem::sources::MomentTensor<Geocentric>;
using CartesianForce = specfem::sources::ForceVector<Cartesian>;
using GeocentricForce = specfem::sources::ForceVector<Geocentric>;

constexpr double deg2rad = specfem::constants::degrees_to_radians_double;
constexpr double tol = 1e-5;

// Colatitude/longitude (radians) from geographic latitude/longitude (degrees).
std::pair<double, double> colat_lon(double lat_deg, double lon_deg) {
  return { std::numbers::pi / 2.0 - lat_deg * deg2rad, lon_deg * deg2rad };
}

auto to_cartesian(double theta, double phi) {
  return specfem::coordinate_systems::rotation_matrix<Cartesian, Geocentric>(
      theta, phi);
}

auto to_geocentric(double theta, double phi) {
  return specfem::coordinate_systems::rotation_matrix<Geocentric, Cartesian>(
      theta, phi);
}

// Flat regional mesh: x = East = e_phi, y = North = -e_theta, z = Up = e_r,
// i.e. the globe rotation at theta = 0, phi = -pi/2 (mirrors
// assembly::sources_impl::flat_mesh_rotation).
auto flat_mesh_rotation() { return to_cartesian(0.0, -std::numbers::pi / 2.0); }

// Build a geocentric tensor from CMTSOLUTION components. Slots are
// (theta, phi, r): (0,0)=Mtt (1,1)=Mpp (2,2)=Mrr (0,1)=Mtp (0,2)=Mrt (1,2)=Mrp.
GeocentricTensor from_cmt(double Mrr, double Mtt, double Mpp, double Mrt,
                          double Mrp, double Mtp) {
  return GeocentricTensor(
      static_cast<type_real>(Mtt), static_cast<type_real>(Mpp),
      static_cast<type_real>(Mrr), static_cast<type_real>(Mtp),
      static_cast<type_real>(Mrt), static_cast<type_real>(Mrp));
}

GeocentricForce from_enu(double east, double north, double up) {
  const auto local = specfem::coordinate_systems::to_geocentric_components(
      specfem::coordinate_systems::EastNorthUp<double>{ east, north, up });
  return GeocentricForce(static_cast<type_real>(local[0]),
                         static_cast<type_real>(local[1]),
                         static_cast<type_real>(local[2]));
}

// Scalar invariants of a 3x3 tensor: trace, second invariant, determinant.
// Equal invariants <=> equal eigenvalues, so preserving all three proves the
// rotation is a similarity.
struct Invariants {
  double trace, second, det;
};

template <typename Frame>
Invariants invariants(const specfem::sources::MomentTensor<Frame> &m) {
  const double trace = m(0, 0) + m(1, 1) + m(2, 2);
  const double second = m(0, 0) * m(1, 1) + m(1, 1) * m(2, 2) +
                        m(2, 2) * m(0, 0) - m(0, 1) * m(1, 0) -
                        m(1, 2) * m(2, 1) - m(0, 2) * m(2, 0);
  const double det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
                     m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
                     m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
  return { trace, second, det };
}

template <typename Frame>
void expect_tensor_near(const specfem::sources::MomentTensor<Frame> &actual,
                        const specfem::sources::MomentTensor<Frame> &expected) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(actual(i, j), expected(i, j), tol)
          << "(" << i << "," << j << ")";
    }
  }
}

double norm(const CartesianForce &f) {
  return std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
}

// An explosion (Mrr=Mtt=Mpp, no shear) is isotropic and must map to the same
// isotropic Cartesian tensor at every latitude and longitude. This is the
// invariance test that catches most basis/sign errors.
TEST(MomentTensorFrameRotation, ExplosionIsotropicEverywhere) {
  const double M0 = 2.5;
  const auto iso = from_cmt(M0, M0, M0, 0.0, 0.0, 0.0);
  const CartesianTensor expected(M0, M0, M0, 0.0, 0.0, 0.0);

  for (double lat = -80.0; lat <= 80.0; lat += 20.0) {
    for (double lon = 0.0; lon <= 360.0; lon += 30.0) {
      const auto [theta, phi] = colat_lon(lat, lon);
      SCOPED_TRACE("lat=" + std::to_string(lat) +
                   " lon=" + std::to_string(lon));
      expect_tensor_near(
          specfem::sources::rotate(iso, to_cartesian(theta, phi)), expected);
    }
  }
}

// A pure Mrt mechanism at the North Pole (theta=0, phi=0) maps to Mxz = Mrt
// with every other Cartesian component zero (e_r = z, e_theta = x, e_phi = y).
TEST(MomentTensorFrameRotation, PureMrtAtNorthPole) {
  const auto m = from_cmt(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
  expect_tensor_near(specfem::sources::rotate(m, to_cartesian(0.0, 0.0)),
                     CartesianTensor(0.0, 0.0, 0.0, 0.0, 1.0, 0.0));
}

// Geocentric -> Cartesian -> geocentric recovers the input to round-off.
TEST(MomentTensorFrameRotation, RoundTripRecoversInput) {
  const auto m = from_cmt(1.0, -2.0, 3.0, 0.5, -1.5, 2.5);
  for (double lat = -70.0; lat <= 70.0; lat += 35.0) {
    for (double lon = 15.0; lon <= 345.0; lon += 55.0) {
      const auto [theta, phi] = colat_lon(lat, lon);
      SCOPED_TRACE("lat=" + std::to_string(lat) +
                   " lon=" + std::to_string(lon));
      const auto cartesian =
          specfem::sources::rotate(m, to_cartesian(theta, phi));
      expect_tensor_near(
          specfem::sources::rotate(cartesian, to_geocentric(theta, phi)), m);
    }
  }
}

// Trace, second invariant, and determinant are preserved by the rotation.
TEST(MomentTensorFrameRotation, InvariantsPreserved) {
  const auto m = from_cmt(4.0, -1.0, 2.0, 0.7, 1.3, -0.4);
  const auto in = invariants(m);
  for (double lat = -60.0; lat <= 60.0; lat += 30.0) {
    for (double lon = 10.0; lon <= 350.0; lon += 85.0) {
      const auto [theta, phi] = colat_lon(lat, lon);
      const auto out =
          invariants(specfem::sources::rotate(m, to_cartesian(theta, phi)));
      SCOPED_TRACE("lat=" + std::to_string(lat) +
                   " lon=" + std::to_string(lon));
      EXPECT_NEAR(out.trace, in.trace, tol);
      EXPECT_NEAR(out.second, in.second, tol);
      EXPECT_NEAR(out.det, in.det, tol);
    }
  }
}

// An asymmetric (Cosserat) tensor keeps its antisymmetric part under the
// rank-2 rotation; the rotation is not restricted to symmetric input.
TEST(MomentTensorFrameRotation, AsymmetricPartSurvives) {
  const CartesianTensor m(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0);
  const auto [theta, phi] = colat_lon(23.0, 47.0);
  const auto local = specfem::sources::rotate(m, to_geocentric(theta, phi));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(local(i, j), -local(j, i), tol);
    }
  }
  expect_tensor_near(specfem::sources::rotate(local, to_cartesian(theta, phi)),
                     m);
}

// The flat-mesh rotation reproduces the regional CMTSOLUTION relabel,
// including the Mxy = -Mtp and Myz = -Mrt signs.
TEST(MomentTensorFrameRotation, FlatMeshReproducesRegionalRelabel) {
  const double Mrr = 1, Mtt = 2, Mpp = 3, Mrt = 4, Mrp = 5, Mtp = 6;
  const auto c = specfem::sources::rotate(
      from_cmt(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp), flat_mesh_rotation());
  EXPECT_NEAR(c(0, 0), Mpp, tol);  // Mxx
  EXPECT_NEAR(c(1, 1), Mtt, tol);  // Myy
  EXPECT_NEAR(c(2, 2), Mrr, tol);  // Mzz
  EXPECT_NEAR(c(0, 1), -Mtp, tol); // Mxy
  EXPECT_NEAR(c(0, 2), Mrp, tol);  // Mxz
  EXPECT_NEAR(c(1, 2), -Mrt, tol); // Myz
  EXPECT_NEAR(c(1, 0), c(0, 1), tol);
  EXPECT_NEAR(c(2, 0), c(0, 2), tol);
  EXPECT_NEAR(c(2, 1), c(1, 2), tol);
}

// At the North Pole (theta=0, phi=0): e_r = +z, e_theta = +x, e_phi = +y, so
// East -> +y, North (= -e_theta) -> -x, Up -> +z.
TEST(ForceFrameRotation, NorthPoleAxes) {
  const auto R = to_cartesian(0.0, 0.0);
  const auto east = specfem::sources::rotate(from_enu(1, 0, 0), R);
  EXPECT_NEAR(east[0], 0.0, tol);
  EXPECT_NEAR(east[1], 1.0, tol);
  EXPECT_NEAR(east[2], 0.0, tol);

  const auto north = specfem::sources::rotate(from_enu(0, 1, 0), R);
  EXPECT_NEAR(north[0], -1.0, tol);
  EXPECT_NEAR(north[1], 0.0, tol);
  EXPECT_NEAR(north[2], 0.0, tol);

  const auto up = specfem::sources::rotate(from_enu(0, 0, 1), R);
  EXPECT_NEAR(up[0], 0.0, tol);
  EXPECT_NEAR(up[1], 0.0, tol);
  EXPECT_NEAR(up[2], 1.0, tol);
}

// On the equator at longitude 0 (theta=pi/2, phi=0): e_r = +x, so Up -> +x,
// North (= -e_theta) -> +z, East (= e_phi) -> +y.
TEST(ForceFrameRotation, EquatorLongitudeZeroAxes) {
  const auto [theta, phi] = colat_lon(0.0, 0.0);
  const auto R = to_cartesian(theta, phi);

  const auto up = specfem::sources::rotate(from_enu(0, 0, 1), R);
  EXPECT_NEAR(up[0], 1.0, tol);
  EXPECT_NEAR(up[1], 0.0, tol);
  EXPECT_NEAR(up[2], 0.0, tol);

  const auto north = specfem::sources::rotate(from_enu(0, 1, 0), R);
  EXPECT_NEAR(north[0], 0.0, tol);
  EXPECT_NEAR(north[1], 0.0, tol);
  EXPECT_NEAR(north[2], 1.0, tol);

  const auto east = specfem::sources::rotate(from_enu(1, 0, 0), R);
  EXPECT_NEAR(east[0], 0.0, tol);
  EXPECT_NEAR(east[1], 1.0, tol);
  EXPECT_NEAR(east[2], 0.0, tol);
}

// A rotation preserves the vector's magnitude at every latitude and longitude.
TEST(ForceFrameRotation, MagnitudePreserved) {
  const auto f = from_enu(1.0, -2.0, 3.0);
  const double n0 = std::sqrt(1.0 + 4.0 + 9.0);
  for (double lat = -80.0; lat <= 80.0; lat += 20.0) {
    for (double lon = 0.0; lon <= 360.0; lon += 45.0) {
      const auto [theta, phi] = colat_lon(lat, lon);
      SCOPED_TRACE("lat=" + std::to_string(lat) +
                   " lon=" + std::to_string(lon));
      EXPECT_NEAR(norm(specfem::sources::rotate(f, to_cartesian(theta, phi))),
                  n0, tol);
    }
  }
}

// On a flat mesh an East/North/Up force passes through as (x, y, z).
TEST(ForceFrameRotation, FlatMeshIsIdentityOnEastNorthUp) {
  const auto c =
      specfem::sources::rotate(from_enu(1, 2, 3), flat_mesh_rotation());
  EXPECT_NEAR(c[0], 1.0, tol);
  EXPECT_NEAR(c[1], 2.0, tol);
  EXPECT_NEAR(c[2], 3.0, tol);
}

} // namespace specfem::sources::frame_rotation_test_impl
