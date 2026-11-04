#include "fk_injection/config.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/utilities/complex_matrix.hpp"
#include "impl/field_recovery.hpp"
#include "impl/fk_math_impl.hpp"
#include "impl/halfspace_solve.hpp"
#include "impl/layer_operators.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <complex>

// ============================================================================
// Helper: build a standard elastic half-space + one finite layer setup.
//
//   hs   : rho=3300, vp=8100, vs=4500 (half-space)
//   layer: rho=2800, vp=6000, vs=3500, H=10000 m
//   p    : sin(20 deg) / 8100 s/m
//   omega: 2*pi*0.5 rad/s, incident P wave
// ============================================================================
namespace field_recovery_tests_impl {

using fk_injection::type_real;

inline void make_elastic_model(fk_injection::ElasticIsotropicLayer &hs,
                               fk_injection::ElasticIsotropicLayer &layer,
                               type_real &p, std::complex<double> &omega,
                               fk_injection::ComplexMatrix<4> &e_mat,
                               fk_injection::ComplexVector<4> &modes) {
  hs.density = static_cast<type_real>(3300.0);
  hs.p_velocity = static_cast<type_real>(8100.0);
  hs.s_velocity = static_cast<type_real>(4500.0);
  hs.thickness = static_cast<type_real>(0.0);

  layer.density = static_cast<type_real>(2800.0);
  layer.p_velocity = static_cast<type_real>(6000.0);
  layer.s_velocity = static_cast<type_real>(3500.0);
  layer.thickness = static_cast<type_real>(10000.0);

  constexpr double deg20_rad = 20.0 * 3.141592653589793 / 180.0;
  p = static_cast<type_real>(std::sin(deg20_rad) / 8100.0);

  omega = std::complex<double>(2.0 * 3.141592653589793 * 0.5, 0.0);

  e_mat = fk_injection::impl::elastic_halfspace_eigenmatrix(hs, p);

  const fk_injection::incident_wave_type wave_type =
      fk_injection::incident_wave_type::p;
  const auto amp = fk_injection::impl::incident_amplitude(
      hs, wave_type, static_cast<type_real>(1.0), p);

  const auto coeffs =
      fk_injection::impl::halfspace_coefficients(e_mat, wave_type, amp);

  modes = fk_injection::impl::build_bottom_vector(wave_type, amp, coeffs);
}

} // namespace field_recovery_tests_impl

// ============================================================================
// Test 5: HalfspaceZeroDepthEqualsModesProjection
//
// At zz=0, G(0)=I so elastic_state_in_halfspace == E * modes component-wise.
// ============================================================================

TEST(FieldRecovery, HalfspaceZeroDepthEqualsModesProjection) {
  fk_injection::ElasticIsotropicLayer hs;
  fk_injection::ElasticIsotropicLayer layer;
  fk_injection::type_real p;
  std::complex<double> omega;
  fk_injection::ComplexMatrix<4> e_mat;
  fk_injection::ComplexVector<4> modes;
  field_recovery_tests_impl::make_elastic_model(hs, layer, p, omega, e_mat,
                                                modes);

  const auto state = fk_injection::impl::elastic_state_in_halfspace(
      e_mat, hs, omega, p, /*zz=*/0.0, modes);

  const auto expected = e_mat * modes;

  constexpr double kTol = 1.0e-8;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(state[i].real(), expected[i].real(), kTol)
        << "component " << i << " real";
    EXPECT_NEAR(state[i].imag(), expected[i].imag(), kTol)
        << "component " << i << " imag";
  }
}

// ============================================================================
// Test 6: ElasticInterfaceContinuity
//
// elastic_state_in_layer at height=0 returns state_at_bottom unchanged,
// because T(0) = I.
// ============================================================================

TEST(FieldRecovery, ElasticInterfaceContinuity) {
  fk_injection::ElasticIsotropicLayer hs;
  fk_injection::ElasticIsotropicLayer layer;
  fk_injection::type_real p;
  std::complex<double> omega;
  fk_injection::ComplexMatrix<4> e_mat;
  fk_injection::ComplexVector<4> modes;
  field_recovery_tests_impl::make_elastic_model(hs, layer, p, omega, e_mat,
                                                modes);

  fk_injection::ComplexVector<4> state_at_bottom;
  state_at_bottom[0] = std::complex<double>(1.0, 0.5);
  state_at_bottom[1] = std::complex<double>(-0.3, 0.2);
  state_at_bottom[2] = std::complex<double>(0.1, -0.4);
  state_at_bottom[3] = std::complex<double>(0.7, 0.0);

  const auto result = fk_injection::impl::elastic_state_in_layer(
      layer, omega, /*height=*/static_cast<fk_injection::type_real>(0.0), p,
      state_at_bottom);

  constexpr double kTol = 1.0e-9;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(result[i].real(), state_at_bottom[i].real(), kTol)
        << "component " << i << " real";
    EXPECT_NEAR(result[i].imag(), state_at_bottom[i].imag(), kTol)
        << "component " << i << " imag";
  }
}

// ============================================================================
// Test 9: ProjectAcousticPotentialRate
//
// Hand-written state: with P/k = 1+2i, chi_dot = i*p*(P/k) = p*(-2+i), and
// u_x = -i*p^2/rho*(P/k) = p^2/rho*(2-i) with rho the density passed in. u_z
// passes through unchanged.
// ============================================================================

TEST(FieldRecovery, ProjectAcousticPotentialRate) {
  const fk_injection::type_real p =
      static_cast<fk_injection::type_real>(1.0e-4);
  const double pd = static_cast<double>(p);
  const double density_below = 2700.0;

  fk_injection::impl::AcousticState state;
  state.u_z = std::complex<double>(3.0, 4.0);
  state.pressure_over_k = std::complex<double>(1.0, 2.0);

  const fk_injection::impl::SeriesSpectrum series =
      fk_injection::impl::project_acoustic(state, density_below, p);

  constexpr double kTol = 1.0e-15;
  const std::complex<double> chi_dot =
      series.values[fk_injection::impl::series_potential_rate];
  EXPECT_NEAR(chi_dot.real(), -2.0 * pd, kTol);
  EXPECT_NEAR(chi_dot.imag(), pd, kTol);

  const std::complex<double> u_x =
      series.values[fk_injection::impl::series_horizontal];
  EXPECT_NEAR(u_x.real(), 2.0 * pd * pd / density_below, kTol);
  EXPECT_NEAR(u_x.imag(), -pd * pd / density_below, kTol);

  const std::complex<double> u_z =
      series.values[fk_injection::impl::series_vertical];
  EXPECT_NEAR(u_z.real(), 3.0, kTol);
  EXPECT_NEAR(u_z.imag(), 4.0, kTol);
}

// ============================================================================
// HalfspaceStateMatchesPropagator
//
// Inside a uniform medium state(0) = T(h) state(-h), so propagating one mode up
// from zz = -h must return its eigenmatrix column: T(h) E = E G(h). This ties
// the signs in G(zz) to the propagator; each row is judged relative to its own
// scale, since displacement and traction rows differ by ~10^11.
// ============================================================================

TEST(FieldRecovery, HalfspaceStateMatchesPropagator) {
  fk_injection::ElasticIsotropicLayer hs;
  fk_injection::ElasticIsotropicLayer layer;
  fk_injection::type_real p;
  std::complex<double> omega;
  fk_injection::ComplexMatrix<4> e_mat;
  fk_injection::ComplexVector<4> modes;
  field_recovery_tests_impl::make_elastic_model(hs, layer, p, omega, e_mat,
                                                modes);

  const double height = 5000.0;
  constexpr double kRelTol = 1.0e-9;

  for (int mode = 0; mode < 4; ++mode) {
    fk_injection::ComplexVector<4> unit_mode{};
    unit_mode[mode] = std::complex<double>(1.0, 0.0);

    const auto state_below = fk_injection::impl::elastic_state_in_halfspace(
        e_mat, hs, omega, p, -height, unit_mode);
    const auto state_at_top = fk_injection::impl::elastic_state_in_layer(
        hs, omega, height, p, state_below);

    for (int i = 0; i < 4; ++i) {
      EXPECT_LT(std::abs(state_at_top[i] - e_mat(i, mode)) /
                    std::abs(e_mat(i, mode)),
                kRelTol)
          << "mode " << mode << " component " << i;
    }
  }
}

// ============================================================================
// AcousticStateInLayer
//
// At height 0 the state is unchanged; above it, the state is the acoustic
// propagator's leading 2x2 block applied to the bottom state.
// ============================================================================

TEST(FieldRecovery, AcousticStateInLayer) {
  fk_injection::AcousticLayer layer;
  layer.density = static_cast<fk_injection::type_real>(1000.0);
  layer.p_velocity = static_cast<fk_injection::type_real>(1500.0);
  layer.thickness = static_cast<fk_injection::type_real>(4000.0);

  const double p = 3.0e-4;
  const std::complex<double> omega(3.0, 0.0);

  fk_injection::impl::AcousticState bottom;
  bottom.u_z = std::complex<double>(3.0, 4.0);
  bottom.pressure_over_k = std::complex<double>(1.0, 2.0);

  constexpr double kTol = 1.0e-12;

  const auto at_bottom =
      fk_injection::impl::acoustic_state_in_layer(layer, omega, 0.0, p, bottom);
  EXPECT_LT(std::abs(at_bottom.u_z - bottom.u_z), kTol);
  EXPECT_LT(std::abs(at_bottom.pressure_over_k - bottom.pressure_over_k), kTol);

  const double height = 1000.0;
  const auto Q = fk_injection::impl::layer_propagator(layer, omega, height, p);
  const auto above = fk_injection::impl::acoustic_state_in_layer(
      layer, omega, height, p, bottom);
  const std::complex<double> expected_u_z =
      Q(0, 0) * bottom.u_z + Q(0, 1) * bottom.pressure_over_k;
  const std::complex<double> expected_pressure =
      Q(1, 0) * bottom.u_z + Q(1, 1) * bottom.pressure_over_k;
  EXPECT_LT(std::abs(above.u_z - expected_u_z), kTol);
  EXPECT_LT(std::abs(above.pressure_over_k - expected_pressure),
            kTol * std::abs(expected_pressure));
}

// ============================================================================
// ProjectElastic
//
// Hand-written state (dx, dz, txz, tzz) = (1+2i, 3+4i, 5+6i, 7+8i) with
// omega*p = 1 + 0.5i:
//   S0 = -i*dx = 2-i, S1 = dz = 3+4i,
//   S2 = omega*p*tzz = 3+11.5i, S3 = -i*omega*p*txz = 8.5-2i,
//   S4 = omega*p*dx = 2.5i.
// Without stress, S2..S4 stay zero.
// ============================================================================

TEST(FieldRecovery, ProjectElastic) {
  const fk_injection::ComplexVector<4> state = {
    std::complex<double>(1.0, 2.0), std::complex<double>(3.0, 4.0),
    std::complex<double>(5.0, 6.0), std::complex<double>(7.0, 8.0)
  };
  const std::complex<double> omega(2.0, 1.0);
  const double p = 0.5;

  constexpr double kTol = 1.0e-14;

  const auto series =
      fk_injection::impl::project_elastic(state, omega, p, true);
  const auto check = [&series](int slot, double real, double imag) {
    EXPECT_NEAR(series.values[slot].real(), real, kTol) << "slot " << slot;
    EXPECT_NEAR(series.values[slot].imag(), imag, kTol) << "slot " << slot;
  };
  check(fk_injection::impl::series_horizontal, 2.0, -1.0);
  check(fk_injection::impl::series_vertical, 3.0, 4.0);
  check(fk_injection::impl::series_sigma_zz, 3.0, 11.5);
  check(fk_injection::impl::series_sigma_rz, 8.5, -2.0);
  check(fk_injection::impl::series_dx, 0.0, 2.5);

  const auto no_stress =
      fk_injection::impl::project_elastic(state, omega, p, false);
  EXPECT_NEAR(no_stress.values[fk_injection::impl::series_vertical].real(), 3.0,
              kTol);
  for (const int slot :
       { fk_injection::impl::series_sigma_zz,
         fk_injection::impl::series_sigma_rz, fk_injection::impl::series_dx }) {
    EXPECT_NEAR(std::abs(no_stress.values[slot]), 0.0, kTol) << "slot " << slot;
  }
}
