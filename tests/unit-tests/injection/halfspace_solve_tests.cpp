#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/injection/fk/impl/halfspace_solve.hpp"
#include "specfem/injection/fk/impl/layer_operators.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Device kernels are expressed as named functors rather than extended
// __device__ lambdas: nvcc forbids an extended lambda in gtest's private
// TestBody().  File-unique namespace (not anonymous) keeps the unity build
// ODR-safe.  All material constants are inlined at the call site.
namespace halfspace_solve_tests_impl {

struct FreeSurfaceTractionVanishesP {
  Kokkos::View<double *> d_out;
  KOKKOS_FUNCTION void operator()(const int) const {
    // Mantle half-space.
    specfem::injection::fk::ElasticIsotropicLayer hs;
    hs.density = static_cast<type_real>(3380.0);
    hs.p_velocity = static_cast<type_real>(8100.0);
    hs.s_velocity = static_cast<type_real>(4500.0);
    hs.thickness = static_cast<type_real>(0.0);

    const type_real p = static_cast<type_real>(1.0e-4);

    // Incident amplitude.
    const specfem::injection::fk::incident_wave_type wave_type =
        specfem::injection::fk::incident_wave_type::p;
    const auto amp = specfem::injection::fk_impl::incident_amplitude(
        hs, wave_type, static_cast<type_real>(1.0), p);

    // N_mat = E (no layers above the half-space, chain is identity).
    const auto n_mat =
        specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(hs, p);

    // Dummy zero qmat_full (not used for elastic-only path).
    specfem::utilities::ComplexMatrix<4> qmat_zero;

    // Solve for half-space coefficients.
    const auto coeffs = specfem::injection::fk_impl::halfspace_coefficients(
        n_mat, false, qmat_zero, wave_type, amp.c1, amp.c3);

    // Build the bottom state vector.
    const auto bot = specfem::injection::fk_impl::build_bottom_vector(
        wave_type, amp.c1, amp.c3, coeffs);

    // r = N * bot.
    const auto r = n_mat * bot;

    // Traction residual (rows 2,3).
    d_out(0) = Kokkos::abs(r[2]);
    d_out(1) = Kokkos::abs(r[3]);

    // Per-row term-magnitude scale: sum_j |N(i,j)| * |bot[j]|. The traction
    // rows have entries ~1e11, so the residual must be judged relative to
    // the magnitude of the terms being summed (not the O(1) displacement
    // rows) — this measures the cancellation to machine precision.
    double scale2 = 0.0;
    double scale3 = 0.0;
    for (int j = 0; j < 4; ++j) {
      scale2 += Kokkos::abs(n_mat(2, j)) * Kokkos::abs(bot[j]);
      scale3 += Kokkos::abs(n_mat(3, j)) * Kokkos::abs(bot[j]);
    }
    if (scale2 < 1.0e-30)
      scale2 = 1.0;
    if (scale3 < 1.0e-30)
      scale3 = 1.0;
    d_out(2) = scale2;
    d_out(3) = scale3;
  }
};

struct FreeSurfaceTractionVanishesSV {
  Kokkos::View<double *> d_out;
  KOKKOS_FUNCTION void operator()(const int) const {
    specfem::injection::fk::ElasticIsotropicLayer hs;
    hs.density = static_cast<type_real>(3380.0);
    hs.p_velocity = static_cast<type_real>(8100.0);
    hs.s_velocity = static_cast<type_real>(4500.0);
    hs.thickness = static_cast<type_real>(0.0);

    const type_real p = static_cast<type_real>(1.0e-4);

    const specfem::injection::fk::incident_wave_type wave_type =
        specfem::injection::fk::incident_wave_type::sv;
    const auto amp = specfem::injection::fk_impl::incident_amplitude(
        hs, wave_type, static_cast<type_real>(1.0), p);

    const auto n_mat =
        specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(hs, p);

    specfem::utilities::ComplexMatrix<4> qmat_zero;

    const auto coeffs = specfem::injection::fk_impl::halfspace_coefficients(
        n_mat, false, qmat_zero, wave_type, amp.c1, amp.c3);

    const auto bot = specfem::injection::fk_impl::build_bottom_vector(
        wave_type, amp.c1, amp.c3, coeffs);

    const auto r = n_mat * bot;

    d_out(0) = Kokkos::abs(r[2]);
    d_out(1) = Kokkos::abs(r[3]);

    // Per-row term-magnitude scale: sum_j |N(i,j)| * |bot[j]| for the
    // traction rows (entries ~1e11), measuring cancellation to machine
    // precision.
    double scale2 = 0.0;
    double scale3 = 0.0;
    for (int j = 0; j < 4; ++j) {
      scale2 += Kokkos::abs(n_mat(2, j)) * Kokkos::abs(bot[j]);
      scale3 += Kokkos::abs(n_mat(3, j)) * Kokkos::abs(bot[j]);
    }
    if (scale2 < 1.0e-30)
      scale2 = 1.0;
    if (scale3 < 1.0e-30)
      scale3 = 1.0;
    d_out(2) = scale2;
    d_out(3) = scale3;
  }
};

struct BuildBottomVectorP {
  Kokkos::View<double *> d_re;
  Kokkos::View<double *> d_im;
  KOKKOS_FUNCTION void operator()(const int) const {
    // Use known values to check slot assignment exactly.
    const Kokkos::complex<double> c1(0.0, 0.0);
    const Kokkos::complex<double> c3(3.0, 4.0);
    Kokkos::Array<Kokkos::complex<double>, 2> coeffs;
    coeffs[0] = Kokkos::complex<double>(1.0, 2.0);
    coeffs[1] = Kokkos::complex<double>(5.0, 6.0);

    const auto bot = specfem::injection::fk_impl::build_bottom_vector(
        specfem::injection::fk::incident_wave_type::p, c1, c3, coeffs);

    d_re(0) = bot[0].real();
    d_im(0) = bot[0].imag();
    d_re(1) = bot[1].real();
    d_im(1) = bot[1].imag();
    d_re(2) = bot[2].real();
    d_im(2) = bot[2].imag();
    d_re(3) = bot[3].real();
    d_im(3) = bot[3].imag();
  }
};

struct BuildBottomVectorSV {
  Kokkos::View<double *> d_re;
  Kokkos::View<double *> d_im;
  KOKKOS_FUNCTION void operator()(const int) const {
    const Kokkos::complex<double> c1(7.0, 8.0);
    const Kokkos::complex<double> c3(0.0, 0.0);
    Kokkos::Array<Kokkos::complex<double>, 2> coeffs;
    coeffs[0] = Kokkos::complex<double>(1.0, 2.0);
    coeffs[1] = Kokkos::complex<double>(5.0, 6.0);

    const auto bot = specfem::injection::fk_impl::build_bottom_vector(
        specfem::injection::fk::incident_wave_type::sv, c1, c3, coeffs);

    d_re(0) = bot[0].real();
    d_im(0) = bot[0].imag();
    d_re(1) = bot[1].real();
    d_im(1) = bot[1].imag();
    d_re(2) = bot[2].real();
    d_im(2) = bot[2].imag();
    d_re(3) = bot[3].real();
    d_im(3) = bot[3].imag();
  }
};

} // namespace halfspace_solve_tests_impl

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
  // 4 real output scalars: abs(r[2]), abs(r[3]), scale0, scale1.
  Kokkos::View<double *> d_out("out", 4);
  Kokkos::deep_copy(d_out, 0.0);

  Kokkos::parallel_for(
      "free_surface_traction_p", Kokkos::RangePolicy<>(0, 1),
      halfspace_solve_tests_impl::FreeSurfaceTractionVanishesP{ d_out });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h, d_out);

  // Traction residual must cancel to machine precision relative to the
  // traction-row term magnitudes.
  EXPECT_LT(h(0) / h(2), 1.0e-10);
  EXPECT_LT(h(1) / h(3), 1.0e-10);
}

// ============================================================================
// Test 2: FreeSurfaceTractionVanishesSV
//
// Same physical check as Test 1 but for an incident SV wave.
// ============================================================================

TEST(HalfspaceSolve, FreeSurfaceTractionVanishesSV) {
  Kokkos::View<double *> d_out("out_sv", 4);
  Kokkos::deep_copy(d_out, 0.0);

  Kokkos::parallel_for(
      "free_surface_traction_sv", Kokkos::RangePolicy<>(0, 1),
      halfspace_solve_tests_impl::FreeSurfaceTractionVanishesSV{ d_out });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h, d_out);

  EXPECT_LT(h(0) / h(2), 1.0e-10);
  EXPECT_LT(h(1) / h(3), 1.0e-10);
}

// ============================================================================
// Test 3: BuildBottomVectorP
//
// For incident P: bot[0]==0, bot[1]==coeff0, bot[2]==c3, bot[3]==coeff1.
// ============================================================================

TEST(HalfspaceSolve, BuildBottomVectorP) {
  constexpr double kTol = 1.0e-15;

  // 8 real outputs: bot[0..3] real, bot[0..3] imag.
  Kokkos::View<double *> d_re("re_p", 4);
  Kokkos::View<double *> d_im("im_p", 4);

  Kokkos::parallel_for(
      "build_bot_vec_p", Kokkos::RangePolicy<>(0, 1),
      halfspace_solve_tests_impl::BuildBottomVectorP{ d_re, d_im });
  Kokkos::fence();

  auto h_re = Kokkos::create_mirror_view(d_re);
  auto h_im = Kokkos::create_mirror_view(d_im);
  Kokkos::deep_copy(h_re, d_re);
  Kokkos::deep_copy(h_im, d_im);

  // bot[0] == 0.
  EXPECT_NEAR(h_re(0), 0.0, kTol);
  EXPECT_NEAR(h_im(0), 0.0, kTol);

  // bot[1] == coeff0 == (1,2).
  EXPECT_NEAR(h_re(1), 1.0, kTol);
  EXPECT_NEAR(h_im(1), 2.0, kTol);

  // bot[2] == c3 == (3,4).
  EXPECT_NEAR(h_re(2), 3.0, kTol);
  EXPECT_NEAR(h_im(2), 4.0, kTol);

  // bot[3] == coeff1 == (5,6).
  EXPECT_NEAR(h_re(3), 5.0, kTol);
  EXPECT_NEAR(h_im(3), 6.0, kTol);
}

// ============================================================================
// Test 4: BuildBottomVectorSV
//
// For incident SV: bot[0]==c1, bot[1]==coeff0 (reflected SV-down), bot[2]==0,
// bot[3]==coeff1. (This is the physically correct placement; it deviates from
// SPECFEM3D, which erroneously puts coeff0 at bot[2] — see
// injection_plan/specfem3d-sv-fk-bug-report.md.)
// ============================================================================

TEST(HalfspaceSolve, BuildBottomVectorSV) {
  constexpr double kTol = 1.0e-15;

  Kokkos::View<double *> d_re("re_sv", 4);
  Kokkos::View<double *> d_im("im_sv", 4);

  Kokkos::parallel_for(
      "build_bot_vec_sv", Kokkos::RangePolicy<>(0, 1),
      halfspace_solve_tests_impl::BuildBottomVectorSV{ d_re, d_im });
  Kokkos::fence();

  auto h_re = Kokkos::create_mirror_view(d_re);
  auto h_im = Kokkos::create_mirror_view(d_im);
  Kokkos::deep_copy(h_re, d_re);
  Kokkos::deep_copy(h_im, d_im);

  // bot[0] == c1 == (7,8).
  EXPECT_NEAR(h_re(0), 7.0, kTol);
  EXPECT_NEAR(h_im(0), 8.0, kTol);

  // bot[1] == coeff0 == (1,2)  (reflected SV-down, column 1).
  EXPECT_NEAR(h_re(1), 1.0, kTol);
  EXPECT_NEAR(h_im(1), 2.0, kTol);

  // bot[2] == 0  (incident P-up slot, unused for SV incidence).
  EXPECT_NEAR(h_re(2), 0.0, kTol);
  EXPECT_NEAR(h_im(2), 0.0, kTol);

  // bot[3] == coeff1 == (5,6)  (reflected P-down, column 3).
  EXPECT_NEAR(h_re(3), 5.0, kTol);
  EXPECT_NEAR(h_im(3), 6.0, kTol);
}
