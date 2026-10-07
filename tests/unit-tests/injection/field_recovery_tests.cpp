#include "specfem/injection/fk/impl/field_recovery.hpp"
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
namespace field_recovery_tests_impl {

struct TractionSpectrumP {
  Kokkos::View<double *> d_out;
  KOKKOS_FUNCTION void operator()(const int) const {
    // Half-space material (mantle-like).
    specfem::injection::fk::ElasticIsotropicLayer hs;
    hs.density = static_cast<type_real>(3380.0);
    hs.p_velocity = static_cast<type_real>(8100.0);
    hs.s_velocity = static_cast<type_real>(4500.0);
    hs.thickness = static_cast<type_real>(0.0);

    const type_real p = static_cast<type_real>(1.0e-4);

    // Incident P wave.
    const specfem::injection::fk::incident_wave_type wave_type =
        specfem::injection::fk::incident_wave_type::p;
    const auto amp = specfem::injection::fk_impl::incident_amplitude(
        hs, wave_type, static_cast<type_real>(1.0), p);

    const auto e_mat =
        specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(hs, p);

    specfem::utilities::ComplexMatrix<4> qmat_zero; // unused (no fluid)
    const auto coeffs = specfem::injection::fk_impl::halfspace_coefficients(
        e_mat, false, qmat_zero, wave_type, amp.c1, amp.c3);

    const auto bot = specfem::injection::fk_impl::build_bottom_vector(
        wave_type, amp.c1, amp.c3, coeffs);

    // omega = 2*pi * 1 Hz (arbitrary non-zero; must be same scale as p).
    const Kokkos::complex<double> omega(2.0 * 3.141592653589793, 0.0);

    // chain_above = identity (no layers above; ignored for in_halfspace)
    specfem::utilities::ComplexMatrix<4> identity =
        specfem::utilities::ComplexMatrix<4>::identity();

    const auto spectrum = specfem::injection::fk_impl::recover_elastic_point(
        e_mat,
        identity, // chain_above (unused: in_halfspace=true)
        hs,       // point_layer (unused: in_halfspace=true)
        hs,       // halfspace (for vertical slownesses)
        omega, p,
        /*in_halfspace=*/true,
        /*above_all_layers=*/false,
        /*zz=*/static_cast<type_real>(0.0),
        /*height=*/static_cast<type_real>(0.0), bot,
        Kokkos::complex<double>(1.0, 0.0), // stf_coeff
        static_cast<type_real>(0.0),       // xi1
        static_cast<type_real>(0.0),       // xim
        /*compute_stress=*/true);

    d_out(0) = Kokkos::abs(spectrum.values[3]);
    d_out(1) = Kokkos::abs(spectrum.values[4]);

    // Scale each traction residual by |omega*p| times the PRE-cancellation
    // term magnitude of that row of E*bot: sum_k |E(row,k)| * |bot[k]|
    // (~1e11). Normalizing by the cancelled working[2]/[3] (which are ~0)
    // would be meaningless.
    const double op = Kokkos::abs(omega) * static_cast<double>(p);
    double term2 = 0.0;
    double term3 = 0.0;
    for (int k = 0; k < 4; ++k) {
      term2 += Kokkos::abs(e_mat(2, k)) * Kokkos::abs(bot[k]);
      term3 += Kokkos::abs(e_mat(3, k)) * Kokkos::abs(bot[k]);
    }
    double scale2 = op * term2;
    double scale3 = op * term3;
    if (scale2 < 1.0e-30)
      scale2 = 1.0;
    if (scale3 < 1.0e-30)
      scale3 = 1.0;
    d_out(2) = scale2;
    d_out(3) = scale3;
  }
};

struct TractionSpectrumSV {
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

    const auto e_mat =
        specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(hs, p);

    specfem::utilities::ComplexMatrix<4> qmat_zero;
    const auto coeffs = specfem::injection::fk_impl::halfspace_coefficients(
        e_mat, false, qmat_zero, wave_type, amp.c1, amp.c3);

    const auto bot = specfem::injection::fk_impl::build_bottom_vector(
        wave_type, amp.c1, amp.c3, coeffs);

    const Kokkos::complex<double> omega(2.0 * 3.141592653589793, 0.0);

    specfem::utilities::ComplexMatrix<4> identity =
        specfem::utilities::ComplexMatrix<4>::identity();

    const auto spectrum = specfem::injection::fk_impl::recover_elastic_point(
        e_mat, identity, hs, hs, omega, p,
        /*in_halfspace=*/true,
        /*above_all_layers=*/false,
        /*zz=*/static_cast<type_real>(0.0),
        /*height=*/static_cast<type_real>(0.0), bot,
        Kokkos::complex<double>(1.0, 0.0), static_cast<type_real>(0.0),
        static_cast<type_real>(0.0),
        /*compute_stress=*/true);

    d_out(0) = Kokkos::abs(spectrum.values[3]);
    d_out(1) = Kokkos::abs(spectrum.values[4]);

    const double op = Kokkos::abs(omega) * static_cast<double>(p);
    double term2 = 0.0;
    double term3 = 0.0;
    for (int k = 0; k < 4; ++k) {
      term2 += Kokkos::abs(e_mat(2, k)) * Kokkos::abs(bot[k]);
      term3 += Kokkos::abs(e_mat(3, k)) * Kokkos::abs(bot[k]);
    }
    double scale2 = op * term2;
    double scale3 = op * term3;
    if (scale2 < 1.0e-30)
      scale2 = 1.0;
    if (scale3 < 1.0e-30)
      scale3 = 1.0;
    d_out(2) = scale2;
    d_out(3) = scale3;
  }
};

struct DisplacementNonzeroP {
  Kokkos::View<double *> d_out;
  KOKKOS_FUNCTION void operator()(const int) const {
    specfem::injection::fk::ElasticIsotropicLayer hs;
    hs.density = static_cast<type_real>(3380.0);
    hs.p_velocity = static_cast<type_real>(8100.0);
    hs.s_velocity = static_cast<type_real>(4500.0);
    hs.thickness = static_cast<type_real>(0.0);

    const type_real p = static_cast<type_real>(1.0e-4);

    const specfem::injection::fk::incident_wave_type wave_type =
        specfem::injection::fk::incident_wave_type::p;
    const auto amp = specfem::injection::fk_impl::incident_amplitude(
        hs, wave_type, static_cast<type_real>(1.0), p);

    const auto e_mat =
        specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(hs, p);

    specfem::utilities::ComplexMatrix<4> qmat_zero;
    const auto coeffs = specfem::injection::fk_impl::halfspace_coefficients(
        e_mat, false, qmat_zero, wave_type, amp.c1, amp.c3);

    const auto bot = specfem::injection::fk_impl::build_bottom_vector(
        wave_type, amp.c1, amp.c3, coeffs);

    const Kokkos::complex<double> omega(2.0 * 3.141592653589793, 0.0);

    specfem::utilities::ComplexMatrix<4> identity =
        specfem::utilities::ComplexMatrix<4>::identity();

    const auto spectrum = specfem::injection::fk_impl::recover_elastic_point(
        e_mat, identity, hs, hs, omega, p,
        /*in_halfspace=*/true,
        /*above_all_layers=*/false,
        /*zz=*/static_cast<type_real>(0.0),
        /*height=*/static_cast<type_real>(0.0), bot,
        Kokkos::complex<double>(1.0, 0.0), static_cast<type_real>(0.0),
        static_cast<type_real>(0.0),
        /*compute_stress=*/false);

    d_out(0) = Kokkos::abs(spectrum.values[0]);
    d_out(1) = Kokkos::abs(spectrum.values[1]);
  }
};

struct DisplacementNonzeroSV {
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

    const auto e_mat =
        specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(hs, p);

    specfem::utilities::ComplexMatrix<4> qmat_zero;
    const auto coeffs = specfem::injection::fk_impl::halfspace_coefficients(
        e_mat, false, qmat_zero, wave_type, amp.c1, amp.c3);

    const auto bot = specfem::injection::fk_impl::build_bottom_vector(
        wave_type, amp.c1, amp.c3, coeffs);

    const Kokkos::complex<double> omega(2.0 * 3.141592653589793, 0.0);

    specfem::utilities::ComplexMatrix<4> identity =
        specfem::utilities::ComplexMatrix<4>::identity();

    const auto spectrum = specfem::injection::fk_impl::recover_elastic_point(
        e_mat, identity, hs, hs, omega, p,
        /*in_halfspace=*/true,
        /*above_all_layers=*/false,
        /*zz=*/static_cast<type_real>(0.0),
        /*height=*/static_cast<type_real>(0.0), bot,
        Kokkos::complex<double>(1.0, 0.0), static_cast<type_real>(0.0),
        static_cast<type_real>(0.0),
        /*compute_stress=*/false);

    d_out(0) = Kokkos::abs(spectrum.values[0]);
    d_out(1) = Kokkos::abs(spectrum.values[1]);
  }
};

} // namespace field_recovery_tests_impl

// ============================================================================
// Test 1: ElasticHalfSpaceFreeSurfaceTractionSpectrum (P incidence)
//
// Single elastic half-space: rho=3380 kg/m^3, vp=8100 m/s, vs=4500 m/s.
// Point is at zz=0 (top of the half-space / free surface), in_halfspace=true.
// At zz=0 the exponential diagonal G is the identity, so working = E * bot_vec.
// The half-space solve already set (E*bot)[2] == (E*bot)[3] == 0, so
//   values[3] = stf * om * p * working[2] * (-i) should vanish,
//   values[4] = stf * om * p * working[3] should vanish.
// We verify relative to the term-magnitude scale of those traction rows.
// ============================================================================

TEST(FieldRecovery, ElasticHalfSpaceFreeSurfaceTractionSpectrumP) {
  // 4 real outputs: |values[3]|, |values[4]|, scale_txz, scale_tzz.
  Kokkos::View<double *> d_out("out_traction_p", 4);
  Kokkos::deep_copy(d_out, 0.0);

  Kokkos::parallel_for("field_recovery_traction_p", Kokkos::RangePolicy<>(0, 1),
                       field_recovery_tests_impl::TractionSpectrumP{ d_out });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h, d_out);

  // At zz=0 the traction components must vanish to machine precision relative
  // to the term-magnitude scale (the halfspace solve zeros exactly those rows).
  EXPECT_LT(h(0) / h(2), 1.0e-10);
  EXPECT_LT(h(1) / h(3), 1.0e-10);
}

// ============================================================================
// Test 2: ElasticHalfSpaceFreeSurfaceTractionSpectrum (SV incidence)
// ============================================================================

TEST(FieldRecovery, ElasticHalfSpaceFreeSurfaceTractionSpectrumSV) {
  Kokkos::View<double *> d_out("out_traction_sv", 4);
  Kokkos::deep_copy(d_out, 0.0);

  Kokkos::parallel_for("field_recovery_traction_sv",
                       Kokkos::RangePolicy<>(0, 1),
                       field_recovery_tests_impl::TractionSpectrumSV{ d_out });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h, d_out);

  EXPECT_LT(h(0) / h(2), 1.0e-10);
  EXPECT_LT(h(1) / h(3), 1.0e-10);
}

// ============================================================================
// Test 3: ElasticHalfSpaceDisplacementNonzeroP
//
// Sanity check: a P-incident free-surface solution must produce at least one
// non-zero displacement component at the surface.  If both values[0] and
// values[1] are zero, the solver produced a degenerate (all-zero) wavefield.
// ============================================================================

TEST(FieldRecovery, ElasticHalfSpaceDisplacementNonzeroP) {
  Kokkos::View<double *> d_out("out_displ_p", 2);
  Kokkos::deep_copy(d_out, 0.0);

  Kokkos::parallel_for(
      "field_recovery_displ_p", Kokkos::RangePolicy<>(0, 1),
      field_recovery_tests_impl::DisplacementNonzeroP{ d_out });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h, d_out);

  // At least one displacement component must be non-trivially nonzero.
  EXPECT_GT(h(0) + h(1), 1.0e-20);
}

// ============================================================================
// Test 4: ElasticHalfSpaceDisplacementNonzeroSV
// ============================================================================

TEST(FieldRecovery, ElasticHalfSpaceDisplacementNonzeroSV) {
  Kokkos::View<double *> d_out("out_displ_sv", 2);
  Kokkos::deep_copy(d_out, 0.0);

  Kokkos::parallel_for(
      "field_recovery_displ_sv", Kokkos::RangePolicy<>(0, 1),
      field_recovery_tests_impl::DisplacementNonzeroSV{ d_out });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h, d_out);

  EXPECT_GT(h(0) + h(1), 1.0e-20);
}
