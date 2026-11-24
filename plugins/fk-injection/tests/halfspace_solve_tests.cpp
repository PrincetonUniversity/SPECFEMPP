#include "fk_injection/config.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/utilities/register_array.hpp"
#include "impl/fk_math_impl.hpp"
#include "impl/halfspace_solve.hpp"
#include "impl/layer_operators.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>

// ============================================================================
// Test 1: FreeSurfaceTractionVanishesP
//
// For a single elastic half-space (no layers above), the propagated N matrix
// equals E = elastic_halfspace_eigenmatrix.  The free-surface condition
// requires zero traction at the surface, so (N * bot)[2] == (N * bot)[3] == 0.
//
// Tolerance rationale: the traction rows of N have entries ~O(2*mu*eta/p)
// ~O(2 * 3380*4500^2 * 1.2e-4 / 1e-4) ~ O(8e11).  Floating-point residuals
// land around 1e-1 in absolute terms, giving a relative residual of roughly
// 1e-1 / 8e11 ~ 1e-13.  We accept < 1e-10 relative to the max displacement-row
// magnitude, which is conservative by ~3 orders.
// ============================================================================

TEST(HalfspaceSolve, FreeSurfaceTractionVanishesP) {
  // Mantle half-space.
  fk_injection::ElasticIsotropicLayer hs;
  hs.density = static_cast<fk_injection::type_real>(3380.0);
  hs.p_velocity = static_cast<fk_injection::type_real>(8100.0);
  hs.s_velocity = static_cast<fk_injection::type_real>(4500.0);
  hs.thickness = static_cast<fk_injection::type_real>(0.0);

  const fk_injection::type_real p =
      static_cast<fk_injection::type_real>(1.0e-4);

  // Incident amplitude.
  const fk_injection::incident_wave_type wave_type =
      fk_injection::incident_wave_type::p;
  const auto amp = fk_injection::impl::incident_amplitude(
      hs, wave_type, static_cast<fk_injection::type_real>(1.0), p);

  // N_mat = E (no layers above the half-space, chain is identity).
  const auto n_mat = fk_injection::impl::elastic_halfspace_eigenmatrix(hs, p);

  // Solve for half-space coefficients.
  const auto coeffs =
      fk_injection::impl::halfspace_coefficients(n_mat, wave_type, amp);

  // Build the bottom state vector.
  const auto bot =
      fk_injection::impl::build_bottom_vector(wave_type, amp, coeffs);

  // r = N * bot.
  const auto r = n_mat * bot;

  // Per-row term-magnitude scale: sum_j |N(i,j)| * |bot[j]|. The traction
  // rows have entries ~1e11, so the residual must be judged relative to
  // the magnitude of the terms being summed (not the O(1) displacement
  // rows) — this measures the cancellation to machine precision.
  double scale2 = 0.0;
  double scale3 = 0.0;
  for (int j = 0; j < 4; ++j) {
    scale2 += std::abs(n_mat(2, j)) * std::abs(bot(j));
    scale3 += std::abs(n_mat(3, j)) * std::abs(bot(j));
  }
  if (scale2 < 1.0e-30)
    scale2 = 1.0;
  if (scale3 < 1.0e-30)
    scale3 = 1.0;

  // Traction residual must cancel to machine precision relative to the
  // traction-row term magnitudes.
  EXPECT_LT(std::abs(r(2)) / scale2, 1.0e-10);
  EXPECT_LT(std::abs(r(3)) / scale3, 1.0e-10);
}

// ============================================================================
// Test 2: FreeSurfaceTractionVanishesSV
//
// Same physical check as Test 1 but for an incident SV wave.
// ============================================================================

TEST(HalfspaceSolve, FreeSurfaceTractionVanishesSV) {
  fk_injection::ElasticIsotropicLayer hs;
  hs.density = static_cast<fk_injection::type_real>(3380.0);
  hs.p_velocity = static_cast<fk_injection::type_real>(8100.0);
  hs.s_velocity = static_cast<fk_injection::type_real>(4500.0);
  hs.thickness = static_cast<fk_injection::type_real>(0.0);

  const fk_injection::type_real p =
      static_cast<fk_injection::type_real>(1.0e-4);

  const fk_injection::incident_wave_type wave_type =
      fk_injection::incident_wave_type::sv;
  const auto amp = fk_injection::impl::incident_amplitude(
      hs, wave_type, static_cast<fk_injection::type_real>(1.0), p);

  const auto n_mat = fk_injection::impl::elastic_halfspace_eigenmatrix(hs, p);

  const auto coeffs =
      fk_injection::impl::halfspace_coefficients(n_mat, wave_type, amp);

  const auto bot =
      fk_injection::impl::build_bottom_vector(wave_type, amp, coeffs);

  const auto r = n_mat * bot;

  // Per-row term-magnitude scale: sum_j |N(i,j)| * |bot[j]| for the
  // traction rows (entries ~1e11), measuring cancellation to machine
  // precision.
  double scale2 = 0.0;
  double scale3 = 0.0;
  for (int j = 0; j < 4; ++j) {
    scale2 += std::abs(n_mat(2, j)) * std::abs(bot(j));
    scale3 += std::abs(n_mat(3, j)) * std::abs(bot(j));
  }
  if (scale2 < 1.0e-30)
    scale2 = 1.0;
  if (scale3 < 1.0e-30)
    scale3 = 1.0;

  EXPECT_LT(std::abs(r(2)) / scale2, 1.0e-10);
  EXPECT_LT(std::abs(r(3)) / scale3, 1.0e-10);
}

// ============================================================================
// Test 3: BuildBottomVectorP
//
// For incident P: bot[0]==0, bot[1]==coeff0, bot[2]==amplitude, bot[3]==coeff1.
// ============================================================================

TEST(HalfspaceSolve, BuildBottomVectorP) {
  constexpr double kTol = 1.0e-15;

  // Use known values to check slot assignment exactly.
  const std::complex<double> amplitude(3.0, 4.0);
  std::array<std::complex<double>, 2> coeffs;
  coeffs[0] = std::complex<double>(1.0, 2.0);
  coeffs[1] = std::complex<double>(5.0, 6.0);

  const auto bot = fk_injection::impl::build_bottom_vector(
      fk_injection::incident_wave_type::p, amplitude, coeffs);

  // bot[0] == 0.
  EXPECT_NEAR(bot(0).real(), 0.0, kTol);
  EXPECT_NEAR(bot(0).imag(), 0.0, kTol);

  // bot[1] == coeff0 == (1,2).
  EXPECT_NEAR(bot(1).real(), 1.0, kTol);
  EXPECT_NEAR(bot(1).imag(), 2.0, kTol);

  // bot[2] == amplitude == (3,4).
  EXPECT_NEAR(bot(2).real(), 3.0, kTol);
  EXPECT_NEAR(bot(2).imag(), 4.0, kTol);

  // bot[3] == coeff1 == (5,6).
  EXPECT_NEAR(bot(3).real(), 5.0, kTol);
  EXPECT_NEAR(bot(3).imag(), 6.0, kTol);
}

// ============================================================================
// Test 4: BuildBottomVectorSV
//
// For incident SV: bot[0]==amplitude, bot[1]==coeff0 (reflected SV-down),
// bot[2]==0, bot[3]==coeff1. (This is the physically correct placement; it
// deviates from SPECFEM3D, which erroneously puts coeff0 at bot[2] — see
// docs/specfem3d-sv-fk-bug-report.md.)
// ============================================================================

TEST(HalfspaceSolve, BuildBottomVectorSV) {
  constexpr double kTol = 1.0e-15;

  const std::complex<double> amplitude(7.0, 8.0);
  std::array<std::complex<double>, 2> coeffs;
  coeffs[0] = std::complex<double>(1.0, 2.0);
  coeffs[1] = std::complex<double>(5.0, 6.0);

  const auto bot = fk_injection::impl::build_bottom_vector(
      fk_injection::incident_wave_type::sv, amplitude, coeffs);

  // bot[0] == amplitude == (7,8).
  EXPECT_NEAR(bot(0).real(), 7.0, kTol);
  EXPECT_NEAR(bot(0).imag(), 8.0, kTol);

  // bot[1] == coeff0 == (1,2)  (reflected SV-down, column 1).
  EXPECT_NEAR(bot(1).real(), 1.0, kTol);
  EXPECT_NEAR(bot(1).imag(), 2.0, kTol);

  // bot[2] == 0  (incident P-up slot, unused for SV incidence).
  EXPECT_NEAR(bot(2).real(), 0.0, kTol);
  EXPECT_NEAR(bot(2).imag(), 0.0, kTol);

  // bot[3] == coeff1 == (5,6)  (reflected P-down, column 3).
  EXPECT_NEAR(bot(3).real(), 5.0, kTol);
  EXPECT_NEAR(bot(3).imag(), 6.0, kTol);
}

// ============================================================================
// Test 5: FreeSurfaceConditionWithFluid
//
// With a fluid layer over the half-space, r = N * bot must have zero shear
// traction at the fluid base (r[2] == 0) and zero pressure at the free surface
// (q10*r[1] - q11*r[3] == 0, row 1 of the fluid propagator applied to the base
// state). Each is judged relative to the magnitude of its terms.
// ============================================================================

TEST(HalfspaceSolve, FreeSurfaceConditionWithFluid) {
  fk_injection::ElasticIsotropicLayer hs;
  hs.density = static_cast<fk_injection::type_real>(3380.0);
  hs.p_velocity = static_cast<fk_injection::type_real>(8100.0);
  hs.s_velocity = static_cast<fk_injection::type_real>(4500.0);

  fk_injection::AcousticLayer fluid;
  fluid.density = static_cast<fk_injection::type_real>(1020.0);
  fluid.p_velocity = static_cast<fk_injection::type_real>(1500.0);
  fluid.thickness = static_cast<fk_injection::type_real>(2000.0);

  const double p = 1.0e-4;
  const std::complex<double> omega(3.0, 0.0);

  const auto n_mat = fk_injection::impl::elastic_halfspace_eigenmatrix(hs, p);
  const auto q_mat = fk_injection::impl::layer_propagator(
      fluid, omega, static_cast<double>(fluid.thickness), p);

  for (const auto wave_type : { fk_injection::incident_wave_type::p,
                                fk_injection::incident_wave_type::sv }) {
    const auto amp = fk_injection::impl::incident_amplitude(
        hs, wave_type, static_cast<fk_injection::type_real>(1.0), p);
    const auto coeffs = fk_injection::impl::halfspace_coefficients(
        n_mat, q_mat, wave_type, amp);
    const auto bot =
        fk_injection::impl::build_bottom_vector(wave_type, amp, coeffs);
    const auto r = n_mat * bot;
    const std::complex<double> surface_pressure =
        q_mat(1, 0) * r(1) - q_mat(1, 1) * r(3);

    double shear_scale = 0.0;
    double pressure_scale = 0.0;
    for (int j = 0; j < 4; ++j) {
      shear_scale += std::abs(n_mat(2, j)) * std::abs(bot(j));
      pressure_scale += (std::abs(q_mat(1, 0)) * std::abs(n_mat(1, j)) +
                         std::abs(q_mat(1, 1)) * std::abs(n_mat(3, j))) *
                        std::abs(bot(j));
    }

    EXPECT_LT(std::abs(r(2)) / shear_scale, 1.0e-10);
    EXPECT_LT(std::abs(surface_pressure) / pressure_scale, 1.0e-10);
  }
}

// ============================================================================
// Test 6: SingularSystemThrows
//
// A zero or non-finite determinant has no reflected amplitudes to return, for
// either overload.
// ============================================================================

namespace halfspace_solve_tests_impl {

inline void expect_singular(const fk_injection::ComplexMatrix<4> &n_mat) {
  const fk_injection::incident_wave_type wave_type =
      fk_injection::incident_wave_type::p;
  const std::complex<double> amplitude(1.0, 0.0);
  const fk_injection::ComplexMatrix<2> fluid_propagator(
      [](int i, int j) { return i == j ? 1.0 : 0.0; });

  try {
    fk_injection::impl::halfspace_coefficients(n_mat, wave_type, amplitude);
    ADD_FAILURE() << "elastic overload did not throw";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("singular"), std::string::npos)
        << error.what();
  }

  try {
    fk_injection::impl::halfspace_coefficients(n_mat, fluid_propagator,
                                               wave_type, amplitude);
    ADD_FAILURE() << "fluid overload did not throw";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("singular"), std::string::npos)
        << error.what();
  }
}

} // namespace halfspace_solve_tests_impl

TEST(HalfspaceSolve, SingularSystemThrows) {
  // Every coefficient zero: delta == 0.
  halfspace_solve_tests_impl::expect_singular(fk_injection::ComplexMatrix<4>());

  // NaN coefficients, as an unchecked supercritical slowness would produce.
  const fk_injection::ComplexMatrix<4> not_finite([](int, int) {
    return std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0);
  });
  halfspace_solve_tests_impl::expect_singular(not_finite);
}
