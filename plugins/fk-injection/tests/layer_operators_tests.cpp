#include "fk_injection/layered_model.hpp"
#include "fk_injection/utilities/complex_matrix.hpp"
#include "impl/field_recovery.hpp"
#include "impl/fk_math_impl.hpp"
#include "impl/layer_operators.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <complex>

// ============================================================================
// Test 1: ElasticPropagatorZeroThickness
//
// At H = 0 the raw (unscaled) propagator P satisfies P = (1 - gamma1) * I,
// i.e. P = (1/gamma0) * I.  After multiplying by gamma0 the result is
// the 4x4 identity.
// ============================================================================

TEST(LayerOperators, ElasticPropagatorZeroThickness) {
  constexpr double kTol = 1.0e-10;

  fk_injection::ElasticIsotropicLayer layer;
  layer.density = static_cast<fk_injection::type_real>(2700.0);
  layer.p_velocity = static_cast<fk_injection::type_real>(6000.0);
  layer.s_velocity = static_cast<fk_injection::type_real>(3464.0);
  layer.thickness = static_cast<fk_injection::type_real>(0.0);

  const fk_injection::type_real p =
      static_cast<fk_injection::type_real>(1.0e-4);
  const std::complex<double> omega(6.0, 0.0);

  // Unscaled propagator at H = 0.
  const auto P = fk_injection::impl::layer_propagator(
      layer, omega, static_cast<fk_injection::type_real>(0.0), p);

  // gamma0 * P should equal the identity.
  const double g0 = fk_injection::impl::elastic_gamma0(layer, p);
  const std::complex<double> g0c(g0, 0.0);

  auto scaled = P;
  for (int k = 0; k < 4 * 4; ++k) {
    scaled.data[k] = scaled.data[k] * g0c;
  }

  auto I = fk_injection::ComplexMatrix<4>::identity();
  double max_err = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double e = std::abs(scaled(i, j) - I(i, j));
      if (e > max_err)
        max_err = e;
    }
  }
  EXPECT_LT(max_err, kTol);
}

// ============================================================================
// Test 2: ElasticPropagatorSemigroup
//
// The physical layer transfer matrix is T(H) = gamma0 * P_bar(H) (the driver
// multiplies each layer's unscaled propagator by gamma0). Being a Thomson-
// Haskell transfer, T composes exactly for a uniform layer: T(H) = T(H/2)^2.
// We verify this on the physical transfer with a relative Frobenius tolerance
// (entries span ~10^11, so an absolute tolerance is meaningless here).
// ============================================================================

TEST(LayerOperators, ElasticPropagatorSemigroup) {
  constexpr double kRelTol = 1.0e-9;

  fk_injection::ElasticIsotropicLayer layer;
  layer.density = static_cast<fk_injection::type_real>(2700.0);
  layer.p_velocity = static_cast<fk_injection::type_real>(6000.0);
  layer.s_velocity = static_cast<fk_injection::type_real>(3464.0);
  layer.thickness = static_cast<fk_injection::type_real>(1000.0);

  // p = 1e-4 s/m — subcritical for both P and S:
  //   1/vp^2 - p^2 = 1/36e6 - 1e-8 = 2.67e-8 > 0  (ok)
  //   1/vs^2 - p^2 = 1/12e6 - 1e-8 = 7.2e-8  > 0  (ok)
  const fk_injection::type_real p =
      static_cast<fk_injection::type_real>(1.0e-4);
  const fk_injection::type_real H =
      static_cast<fk_injection::type_real>(1000.0);
  const fk_injection::type_real H2 =
      static_cast<fk_injection::type_real>(500.0);
  const std::complex<double> omega(6.0, 0.0);

  const auto PH = fk_injection::impl::layer_propagator(layer, omega, H, p);
  const auto Ph = fk_injection::impl::layer_propagator(layer, omega, H2, p);

  // Physical transfer matrices T = gamma0 * P_bar.
  const double g0 = fk_injection::impl::elastic_gamma0(layer, p);
  const std::complex<double> g0c(g0, 0.0);

  auto full = PH;
  auto half = Ph;
  for (int k = 0; k < 4 * 4; ++k) {
    full.data[k] = full.data[k] * g0c;
    half.data[k] = half.data[k] * g0c;
  }
  const auto half_squared = half * half;

  // Relative Frobenius norm ||T(H) - T(H/2)^2|| / ||T(H)||.
  double numerator = 0.0;
  double denominator = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const double d = std::abs(full(i, j) - half_squared(i, j));
      const double n = std::abs(full(i, j));
      numerator += d * d;
      denominator += n * n;
    }
  }
  EXPECT_LT(std::sqrt(numerator) / std::sqrt(denominator), kRelTol);
}

// ============================================================================
// Test 3: AcousticPropagatorCoshSinh
//
// For a water layer verify the 2x2 block entries against independently computed
// cosh/sinh expressions.  The (2,2) entry must be zero (block stays zero).
// ============================================================================

TEST(LayerOperators, AcousticPropagatorCoshSinh) {
  constexpr double kTol = 1.0e-10;

  fk_injection::AcousticLayer layer;
  layer.density = static_cast<fk_injection::type_real>(1000.0);
  layer.p_velocity = static_cast<fk_injection::type_real>(1500.0);
  layer.thickness = static_cast<fk_injection::type_real>(4000.0);

  const fk_injection::type_real p =
      static_cast<fk_injection::type_real>(3.0e-4);
  const fk_injection::type_real H =
      static_cast<fk_injection::type_real>(4000.0);
  const std::complex<double> omega(2.0 * fk_injection::impl::pi, 0.0);

  const auto Q = fk_injection::impl::layer_propagator(layer, omega, H, p);

  // Independent reference using the scalar acoustic vertical_slowness.
  const std::complex<double> eta =
      fk_injection::impl::vertical_slowness(layer, p);
  const double rho = static_cast<double>(layer.density);
  const double pd = static_cast<double>(p);
  const double Hd = static_cast<double>(H);

  const std::complex<double> c1 = omega * eta * Hd;
  const std::complex<double> exp_p = std::exp(c1);
  const std::complex<double> exp_m = std::exp(-c1);
  const std::complex<double> ca = (exp_p + exp_m) * 0.5;
  const std::complex<double> sa = (exp_p - exp_m) * 0.5;

  const std::complex<double> ref_01 = -sa * eta * pd / rho;
  const std::complex<double> ref_10 = sa * rho / (pd * eta);

  // Q(0,0) == Q(1,1) == ca.
  EXPECT_NEAR(Q(0, 0).real(), ca.real(), kTol);
  EXPECT_NEAR(Q(0, 0).imag(), ca.imag(), kTol);
  EXPECT_NEAR(Q(1, 1).real(), ca.real(), kTol);
  EXPECT_NEAR(Q(1, 1).imag(), ca.imag(), kTol);

  // Q(0,1) == -sa*eta*p/rho.
  EXPECT_NEAR(Q(0, 1).real(), ref_01.real(), kTol);
  EXPECT_NEAR(Q(0, 1).imag(), ref_01.imag(), kTol);

  // Q(1,0) == sa*rho/(p*eta).
  EXPECT_NEAR(Q(1, 0).real(), ref_10.real(), kTol);
  EXPECT_NEAR(Q(1, 0).imag(), ref_10.imag(), kTol);

  // Q(2,2) == 0 (block stays zero).
  EXPECT_NEAR(Q(2, 2).real(), 0.0, kTol);
  EXPECT_NEAR(Q(2, 2).imag(), 0.0, kTol);
}

// ============================================================================
// Test 4: HalfspaceEigenmatrix
//
// Verify selected entries of E against Tong (2014) A10 analytic expressions.
// ============================================================================

TEST(LayerOperators, HalfspaceEigenmatrix) {
  constexpr double kTol = 1.0e-10;

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = static_cast<fk_injection::type_real>(3380.0);
  hs.p_velocity = static_cast<fk_injection::type_real>(8100.0);
  hs.s_velocity = static_cast<fk_injection::type_real>(4500.0);
  hs.thickness = static_cast<fk_injection::type_real>(0.0);

  const fk_injection::type_real p =
      static_cast<fk_injection::type_real>(1.0e-4);

  const auto E = fk_injection::impl::elastic_halfspace_eigenmatrix(hs, p);

  const auto slowness = fk_injection::impl::vertical_slowness(hs, p);
  const std::complex<double> eta_alpha = slowness[0];
  const std::complex<double> eta_beta = slowness[1];
  const double pd = static_cast<double>(p);
  const double rho = static_cast<double>(hs.density);
  const double vs = static_cast<double>(hs.s_velocity);
  const double two_mul = 2.0 * rho * vs * vs;

  // E(0,2) == 1.
  EXPECT_NEAR(E(0, 2).real(), 1.0, kTol);
  EXPECT_NEAR(E(0, 2).imag(), 0.0, kTol);

  // E(1,0) == 1.
  EXPECT_NEAR(E(1, 0).real(), 1.0, kTol);
  EXPECT_NEAR(E(1, 0).imag(), 0.0, kTol);

  // E(0,0) == eta_beta / p.
  const std::complex<double> ref_00 = eta_beta / pd;
  EXPECT_NEAR(E(0, 0).real(), ref_00.real(), kTol);
  EXPECT_NEAR(E(0, 0).imag(), ref_00.imag(), kTol);

  // E(2,2) == 2*rho*vs^2 * eta_alpha / p.
  const std::complex<double> ref_22 =
      std::complex<double>(two_mul, 0.0) * eta_alpha / pd;
  EXPECT_NEAR(E(2, 2).real(), ref_22.real(), kTol);
  EXPECT_NEAR(E(2, 2).imag(), ref_22.imag(), kTol);
}

// ============================================================================
// Test 5: AcousticFromSolid
//
// Given state {a,b,c,d}: u_z == b and P/k == -d.
// ============================================================================

TEST(LayerOperators, AcousticFromSolid) {
  constexpr double kTol = 1.0e-12;

  fk_injection::ComplexVector<4> state;
  state[0] = std::complex<double>(1.0, 2.0); // a
  state[1] = std::complex<double>(3.0, 4.0); // b
  state[2] = std::complex<double>(5.0, 6.0); // c
  state[3] = std::complex<double>(7.0, 8.0); // d

  const fk_injection::impl::AcousticState result =
      fk_injection::impl::acoustic_from_solid(state);

  // u_z == b == (3,4).
  EXPECT_NEAR(result.u_z.real(), 3.0, kTol);
  EXPECT_NEAR(result.u_z.imag(), 4.0, kTol);

  // P/k == -d == (-7,-8).
  EXPECT_NEAR(result.pressure_over_k.real(), -7.0, kTol);
  EXPECT_NEAR(result.pressure_over_k.imag(), -8.0, kTol);
}
