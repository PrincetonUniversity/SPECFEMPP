#include "specfem/constants.hpp"
#include "specfem/coordinate_systems/spherical_basis.hpp"
#include "specfem/source/moment_tensor_rotation.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>

using specfem::coordinate_systems::local_spherical_basis;
using specfem::sources::CartesianMomentTensor;
using specfem::sources::geographic_to_enu;
using specfem::sources::GeographicMomentTensor;
using specfem::sources::rotate_moment_tensor_geographic_to_cartesian;

namespace {

constexpr double deg2rad = specfem::constants::degrees_to_radians_double;

// Colatitude/longitude (radians) from geographic latitude/longitude (degrees).
std::pair<double, double> colat_lon(double lat_deg, double lon_deg) {
  return { M_PI / 2.0 - lat_deg * deg2rad, lon_deg * deg2rad };
}

// Build a geographic moment tensor from double literals (type_real is float in
// single-precision builds, so brace-init would narrow; cast once here).
GeographicMomentTensor make_geographic(double rr, double tt, double pp,
                                       double rt, double rp, double tp) {
  return { static_cast<type_real>(rr), static_cast<type_real>(tt),
           static_cast<type_real>(pp), static_cast<type_real>(rt),
           static_cast<type_real>(rp), static_cast<type_real>(tp) };
}

double dot(const std::array<double, 3> &a, const std::array<double, 3> &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Scalar invariants of a symmetric 3x3 tensor: trace, second invariant, and
// determinant. Equal invariants <=> equal eigenvalues (same characteristic
// polynomial), so preserving all three proves the rotation is a similarity.
struct Invariants {
  double trace, second, det;
};

Invariants cartesian_invariants(const CartesianMomentTensor &m) {
  const double trace = m.Mxx + m.Myy + m.Mzz;
  const double second = m.Mxx * m.Myy + m.Myy * m.Mzz + m.Mzz * m.Mxx -
                        m.Mxy * m.Mxy - m.Myz * m.Myz - m.Mxz * m.Mxz;
  const double det = m.Mxx * (m.Myy * m.Mzz - m.Myz * m.Myz) -
                     m.Mxy * (m.Mxy * m.Mzz - m.Myz * m.Mxz) +
                     m.Mxz * (m.Mxy * m.Myz - m.Myy * m.Mxz);
  return { trace, second, det };
}

Invariants spherical_invariants(const GeographicMomentTensor &m) {
  const double trace = m.Mrr + m.Mtt + m.Mpp;
  const double second = m.Mrr * m.Mtt + m.Mtt * m.Mpp + m.Mpp * m.Mrr -
                        m.Mrt * m.Mrt - m.Mtp * m.Mtp - m.Mrp * m.Mrp;
  const double det = m.Mrr * (m.Mtt * m.Mpp - m.Mtp * m.Mtp) -
                     m.Mrt * (m.Mrt * m.Mpp - m.Mtp * m.Mrp) +
                     m.Mrp * (m.Mrt * m.Mtp - m.Mtt * m.Mrp);
  return { trace, second, det };
}

// Inverse rotation R^T M_cart R, done in the test from the public basis, to
// check the forward map round-trips and the basis is orthonormal.
GeographicMomentTensor cartesian_to_spherical(double theta, double phi,
                                              const CartesianMomentTensor &c) {
  const auto b = local_spherical_basis(theta, phi);
  const std::array<std::array<double, 3>, 3> r = {
    { { b.e_r[0], b.e_theta[0], b.e_phi[0] },
      { b.e_r[1], b.e_theta[1], b.e_phi[1] },
      { b.e_r[2], b.e_theta[2], b.e_phi[2] } }
  };
  const std::array<std::array<double, 3>, 3> cart = { { { c.Mxx, c.Mxy, c.Mxz },
                                                        { c.Mxy, c.Myy, c.Myz },
                                                        { c.Mxz, c.Myz,
                                                          c.Mzz } } };
  // out = R^T * cart * R
  std::array<std::array<double, 3>, 3> tmp{}, out{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        tmp[i][j] += r[k][i] * cart[k][j];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        out[i][j] += tmp[i][k] * r[k][j];
  // (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp) = diagonal, then (r,theta), (r,phi),
  // (theta,phi).
  return make_geographic(out[0][0], out[1][1], out[2][2], out[0][1], out[0][2],
                         out[1][2]);
}

constexpr double tol = 1e-5;

} // namespace

// An explosion (Mrr=Mtt=Mpp, no shear) is isotropic and must map to the same
// isotropic Cartesian tensor at every latitude and longitude. This is the
// invariance test that catches most basis/sign errors.
TEST(MomentTensorRotation, ExplosionIsotropicEverywhere) {
  const double M0 = 2.5;
  const auto iso = make_geographic(M0, M0, M0, 0.0, 0.0, 0.0);

  for (double lat = -80.0; lat <= 80.0; lat += 20.0) {
    for (double lon = 0.0; lon <= 360.0; lon += 30.0) {
      const auto [theta, phi] = colat_lon(lat, lon);
      const auto c =
          rotate_moment_tensor_geographic_to_cartesian(theta, phi, iso);
      SCOPED_TRACE("lat=" + std::to_string(lat) +
                   " lon=" + std::to_string(lon));
      EXPECT_NEAR(c.Mxx, M0, tol);
      EXPECT_NEAR(c.Myy, M0, tol);
      EXPECT_NEAR(c.Mzz, M0, tol);
      EXPECT_NEAR(c.Mxy, 0.0, tol);
      EXPECT_NEAR(c.Mxz, 0.0, tol);
      EXPECT_NEAR(c.Myz, 0.0, tol);
    }
  }
}

// A pure Mrt mechanism at the North Pole (theta=0, phi=0) maps analytically to
// Mxz = Mrt with every other Cartesian component zero (derived from R M R^T
// with e_r=z, e_theta=x, e_phi=y).
TEST(MomentTensorRotation, PureMrtAtNorthPole) {
  const auto m = make_geographic(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
  const auto c = rotate_moment_tensor_geographic_to_cartesian(0.0, 0.0, m);
  EXPECT_NEAR(c.Mxx, 0.0, tol);
  EXPECT_NEAR(c.Myy, 0.0, tol);
  EXPECT_NEAR(c.Mzz, 0.0, tol);
  EXPECT_NEAR(c.Mxy, 0.0, tol);
  EXPECT_NEAR(c.Mxz, 1.0, tol);
  EXPECT_NEAR(c.Myz, 0.0, tol);
}

// Spherical -> Cartesian -> spherical recovers the input to round-off.
TEST(MomentTensorRotation, RoundTripRecoversInput) {
  const auto m = make_geographic(1.0, -2.0, 3.0, 0.5, -1.5, 2.5);
  for (double lat = -70.0; lat <= 70.0; lat += 35.0) {
    for (double lon = 15.0; lon <= 345.0; lon += 55.0) {
      const auto [theta, phi] = colat_lon(lat, lon);
      const auto c =
          rotate_moment_tensor_geographic_to_cartesian(theta, phi, m);
      const auto back = cartesian_to_spherical(theta, phi, c);
      SCOPED_TRACE("lat=" + std::to_string(lat) +
                   " lon=" + std::to_string(lon));
      EXPECT_NEAR(back.Mrr, m.Mrr, tol);
      EXPECT_NEAR(back.Mtt, m.Mtt, tol);
      EXPECT_NEAR(back.Mpp, m.Mpp, tol);
      EXPECT_NEAR(back.Mrt, m.Mrt, tol);
      EXPECT_NEAR(back.Mrp, m.Mrp, tol);
      EXPECT_NEAR(back.Mtp, m.Mtp, tol);
    }
  }
}

// Trace, second invariant, and determinant are preserved by the rotation
// (orthogonal similarity preserves eigenvalues).
TEST(MomentTensorRotation, InvariantsPreserved) {
  const auto m = make_geographic(4.0, -1.0, 2.0, 0.7, 1.3, -0.4);
  const auto in = spherical_invariants(m);
  for (double lat = -60.0; lat <= 60.0; lat += 30.0) {
    for (double lon = 10.0; lon <= 350.0; lon += 85.0) {
      const auto [theta, phi] = colat_lon(lat, lon);
      const auto c =
          rotate_moment_tensor_geographic_to_cartesian(theta, phi, m);
      const auto out = cartesian_invariants(c);
      SCOPED_TRACE("lat=" + std::to_string(lat) +
                   " lon=" + std::to_string(lon));
      EXPECT_NEAR(out.trace, in.trace, tol);
      EXPECT_NEAR(out.second, in.second, tol);
      EXPECT_NEAR(out.det, in.det, tol);
    }
  }
}

// The local basis is orthonormal at an arbitrary point.
TEST(MomentTensorRotation, BasisOrthonormal) {
  const auto [theta, phi] = colat_lon(23.0, 47.0);
  const auto b = local_spherical_basis(theta, phi);
  EXPECT_NEAR(dot(b.e_r, b.e_r), 1.0, tol);
  EXPECT_NEAR(dot(b.e_theta, b.e_theta), 1.0, tol);
  EXPECT_NEAR(dot(b.e_phi, b.e_phi), 1.0, tol);
  EXPECT_NEAR(dot(b.e_r, b.e_theta), 0.0, tol);
  EXPECT_NEAR(dot(b.e_r, b.e_phi), 0.0, tol);
  EXPECT_NEAR(dot(b.e_theta, b.e_phi), 0.0, tol);
}

// The flat East-North-Up relabel matches the documented regional convention.
TEST(MomentTensorRotation, EnuRelabel) {
  const auto m = make_geographic(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  const auto c = geographic_to_enu(m);
  EXPECT_NEAR(c.Mxx, m.Mpp, tol);  // 3
  EXPECT_NEAR(c.Myy, m.Mtt, tol);  // 2
  EXPECT_NEAR(c.Mzz, m.Mrr, tol);  // 1
  EXPECT_NEAR(c.Mxy, -m.Mtp, tol); // -6
  EXPECT_NEAR(c.Mxz, m.Mrp, tol);  // 5
  EXPECT_NEAR(c.Myz, -m.Mrt, tol); // -4
}
