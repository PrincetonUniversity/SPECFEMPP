#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/injection/fk/impl/layer_operators.hpp"
#include "specfem/injection/fk/impl/medium_interface_coupler.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Device kernels are expressed as named functors rather than extended
// __device__ lambdas: nvcc forbids an extended lambda in gtest's private
// TestBody().  File-unique namespace (not anonymous) keeps the unity build
// ODR-safe.
namespace layer_operators_tests_impl {

struct ElasticPropagatorZeroThickness {
  Kokkos::View<double *> d_err;
  KOKKOS_FUNCTION void operator()(const int) const {
    specfem::injection::fk::ElasticIsotropicLayer layer;
    layer.density = static_cast<type_real>(2700.0);
    layer.p_velocity = static_cast<type_real>(6000.0);
    layer.s_velocity = static_cast<type_real>(3464.0);
    layer.thickness = static_cast<type_real>(0.0);

    const type_real p = static_cast<type_real>(1.0e-4);
    const Kokkos::complex<double> omega(6.0, 0.0);

    // Unscaled propagator at H = 0.
    const auto P = specfem::injection::fk_impl::layer_propagator(
        layer, omega, static_cast<type_real>(0.0), p);

    // gamma0 * P should equal the identity.
    const double g0 = specfem::injection::fk_impl::elastic_gamma0(layer, p);
    const Kokkos::complex<double> g0c(g0, 0.0);

    auto scaled = P;
    for (int k = 0; k < 4 * 4; ++k) {
      scaled.data[k] = scaled.data[k] * g0c;
    }

    auto I = specfem::utilities::ComplexMatrix<4>::identity();
    double max_err = 0.0;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        double e = Kokkos::abs(scaled(i, j) - I(i, j));
        if (e > max_err)
          max_err = e;
      }
    }
    d_err(0) = max_err;
  }
};

struct ElasticPropagatorSemigroup {
  Kokkos::View<double *> d_rel;
  KOKKOS_FUNCTION void operator()(const int) const {
    specfem::injection::fk::ElasticIsotropicLayer layer;
    layer.density = static_cast<type_real>(2700.0);
    layer.p_velocity = static_cast<type_real>(6000.0);
    layer.s_velocity = static_cast<type_real>(3464.0);
    layer.thickness = static_cast<type_real>(1000.0);

    // p = 1e-4 s/m — subcritical for both P and S:
    //   1/vp^2 - p^2 = 1/36e6 - 1e-8 = 2.67e-8 > 0  (ok)
    //   1/vs^2 - p^2 = 1/12e6 - 1e-8 = 7.2e-8  > 0  (ok)
    const type_real p = static_cast<type_real>(1.0e-4);
    const type_real H = static_cast<type_real>(1000.0);
    const type_real H2 = static_cast<type_real>(500.0);
    const Kokkos::complex<double> omega(6.0, 0.0);

    const auto PH =
        specfem::injection::fk_impl::layer_propagator(layer, omega, H, p);
    const auto Ph =
        specfem::injection::fk_impl::layer_propagator(layer, omega, H2, p);

    // Physical transfer matrices T = gamma0 * P_bar.
    const double g0 = specfem::injection::fk_impl::elastic_gamma0(layer, p);
    const Kokkos::complex<double> g0c(g0, 0.0);

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
        const double d = Kokkos::abs(full(i, j) - half_squared(i, j));
        const double n = Kokkos::abs(full(i, j));
        numerator += d * d;
        denominator += n * n;
      }
    }
    d_rel(0) = Kokkos::sqrt(numerator) / Kokkos::sqrt(denominator);
  }
};

struct AcousticPropagatorCoshSinh {
  Kokkos::View<double *> d_re;
  Kokkos::View<double *> d_im;
  Kokkos::View<double *> d_ref_re;
  Kokkos::View<double *> d_ref_im;
  KOKKOS_FUNCTION void operator()(const int) const {
    specfem::injection::fk::AcousticLayer layer;
    layer.density = static_cast<type_real>(1000.0);
    layer.p_velocity = static_cast<type_real>(1500.0);
    layer.thickness = static_cast<type_real>(4000.0);

    const type_real p = static_cast<type_real>(3.0e-4);
    const type_real H = static_cast<type_real>(4000.0);
    const Kokkos::complex<double> omega(2.0 * specfem::injection::fk_impl::pi,
                                        0.0);

    const auto Q =
        specfem::injection::fk_impl::layer_propagator(layer, omega, H, p);

    d_re(0) = Q(0, 0).real();
    d_im(0) = Q(0, 0).imag();
    d_re(1) = Q(1, 1).real();
    d_im(1) = Q(1, 1).imag();
    d_re(2) = Q(0, 1).real();
    d_im(2) = Q(0, 1).imag();
    d_re(3) = Q(1, 0).real();
    d_im(3) = Q(1, 0).imag();
    // Q(2,2) must be zero.
    d_re(4) = Q(2, 2).real();
    d_im(4) = Q(2, 2).imag();

    // Independent reference using the scalar acoustic vertical_slowness.
    const Kokkos::complex<double> eta =
        specfem::injection::fk_impl::vertical_slowness(layer, p);
    const double rho = static_cast<double>(layer.density);
    const double pd = static_cast<double>(p);
    const double Hd = static_cast<double>(H);

    const Kokkos::complex<double> c1 = omega * eta * Hd;
    const Kokkos::complex<double> exp_p = Kokkos::exp(c1);
    const Kokkos::complex<double> exp_m = Kokkos::exp(-c1);
    const Kokkos::complex<double> ca = (exp_p + exp_m) * 0.5;
    const Kokkos::complex<double> sa = (exp_p - exp_m) * 0.5;

    const Kokkos::complex<double> ref_01 = -sa * eta * pd / rho;
    const Kokkos::complex<double> ref_10 = sa * rho / (pd * eta);

    d_ref_re(0) = ca.real();
    d_ref_im(0) = ca.imag();
    d_ref_re(1) = ref_01.real();
    d_ref_im(1) = ref_01.imag();
    d_ref_re(2) = ref_10.real();
    d_ref_im(2) = ref_10.imag();
    d_ref_re(3) = ca.real(); // Q(1,1) == ca as well
    d_ref_im(3) = ca.imag();
  }
};

struct HalfspaceEigenmatrix {
  Kokkos::View<double *> d_re;
  Kokkos::View<double *> d_im;
  Kokkos::View<double *> d_ref_re;
  Kokkos::View<double *> d_ref_im;
  KOKKOS_FUNCTION void operator()(const int) const {
    specfem::injection::fk::ElasticIsotropicLayer hs;
    hs.density = static_cast<type_real>(3380.0);
    hs.p_velocity = static_cast<type_real>(8100.0);
    hs.s_velocity = static_cast<type_real>(4500.0);
    hs.thickness = static_cast<type_real>(0.0);

    const type_real p = static_cast<type_real>(1.0e-4);

    const auto E =
        specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(hs, p);

    // E(0,2) == 1.
    d_re(0) = E(0, 2).real();
    d_im(0) = E(0, 2).imag();

    // E(1,0) == 1.
    d_re(1) = E(1, 0).real();
    d_im(1) = E(1, 0).imag();

    // E(0,0) == eta_beta / p.
    const auto slowness = specfem::injection::fk_impl::vertical_slowness(hs, p);
    const Kokkos::complex<double> eta_alpha = slowness[0];
    const Kokkos::complex<double> eta_beta = slowness[1];
    const double pd = static_cast<double>(p);
    const double rho = static_cast<double>(hs.density);
    const double vs = static_cast<double>(hs.s_velocity);
    const double two_mul = 2.0 * rho * vs * vs;
    const double g1 = specfem::injection::fk_impl::elastic_gamma1(hs, p);

    const Kokkos::complex<double> ref_00 = eta_beta / pd;
    d_re(2) = E(0, 0).real();
    d_im(2) = E(0, 0).imag();
    d_ref_re(0) = ref_00.real();
    d_ref_im(0) = ref_00.imag();

    // E(2,2) == 2*rho*vs^2 * eta_alpha / p.
    const Kokkos::complex<double> ref_22 =
        Kokkos::complex<double>(two_mul, 0.0) * eta_alpha / pd;
    d_re(3) = E(2, 2).real();
    d_im(3) = E(2, 2).imag();
    d_ref_re(1) = ref_22.real();
    d_ref_im(1) = ref_22.imag();

    // Also store constants for E(0,2)==1 and E(1,0)==1.
    d_ref_re(2) = 1.0;
    d_ref_im(2) = 0.0;
    d_ref_re(3) = 1.0;
    d_ref_im(3) = 0.0;
  }
};

struct CoupleSolidToFluid {
  Kokkos::View<double *> d_re;
  Kokkos::View<double *> d_im;
  KOKKOS_FUNCTION void operator()(const int) const {
    specfem::utilities::ComplexVector<4> state;
    state[0] = Kokkos::complex<double>(1.0, 2.0); // a
    state[1] = Kokkos::complex<double>(3.0, 4.0); // b
    state[2] = Kokkos::complex<double>(5.0, 6.0); // c
    state[3] = Kokkos::complex<double>(7.0, 8.0); // d

    const auto result =
        specfem::injection::fk_impl::couple_solid_to_fluid(state);

    d_re(0) = result[0].real();
    d_im(0) = result[0].imag();
    d_re(1) = result[1].real();
    d_im(1) = result[1].imag();
    d_re(2) = result[2].real();
    d_im(2) = result[2].imag();
    d_re(3) = result[3].real();
    d_im(3) = result[3].imag();
  }
};

} // namespace layer_operators_tests_impl

// ============================================================================
// Test 1: ElasticPropagatorZeroThickness
//
// At H = 0 the raw (unscaled) propagator P satisfies P = (1 - gamma1) * I,
// i.e. P = (1/gamma0) * I.  After multiplying by gamma0 the result is
// the 4x4 identity.
// ============================================================================

TEST(LayerOperators, ElasticPropagatorZeroThickness) {
  constexpr double kTol = 1.0e-10;

  Kokkos::View<double *> d_err("err", 1);
  Kokkos::deep_copy(d_err, 0.0);

  Kokkos::parallel_for(
      "elastic_propagator_zero_H", Kokkos::RangePolicy<>(0, 1),
      layer_operators_tests_impl::ElasticPropagatorZeroThickness{ d_err });
  Kokkos::fence();

  auto h_err = Kokkos::create_mirror_view(d_err);
  Kokkos::deep_copy(h_err, d_err);
  EXPECT_LT(h_err(0), kTol);
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

  Kokkos::View<double *> d_rel("rel", 1);
  Kokkos::deep_copy(d_rel, 0.0);

  Kokkos::parallel_for(
      "elastic_propagator_semigroup", Kokkos::RangePolicy<>(0, 1),
      layer_operators_tests_impl::ElasticPropagatorSemigroup{ d_rel });
  Kokkos::fence();

  auto h_rel = Kokkos::create_mirror_view(d_rel);
  Kokkos::deep_copy(h_rel, d_rel);
  EXPECT_LT(h_rel(0), kRelTol);
}

// ============================================================================
// Test 3: AcousticPropagatorCoshSinh
//
// For a water layer verify the 2x2 block entries against independently computed
// cosh/sinh expressions.  The (2,2) entry must be zero (block stays zero).
// ============================================================================

TEST(LayerOperators, AcousticPropagatorCoshSinh) {
  constexpr double kTol = 1.0e-10;

  // 7 scalar outputs: Q(0,0), Q(1,1), Q(0,1) re/im, Q(1,0) re/im, Q(2,2).
  Kokkos::View<double *> d_re("re", 7);
  Kokkos::View<double *> d_im("im", 7);
  Kokkos::View<double *> d_ref_re("ref_re", 4);
  Kokkos::View<double *> d_ref_im("ref_im", 4);

  Kokkos::parallel_for("acoustic_propagator_coshsinh",
                       Kokkos::RangePolicy<>(0, 1),
                       layer_operators_tests_impl::AcousticPropagatorCoshSinh{
                           d_re, d_im, d_ref_re, d_ref_im });
  Kokkos::fence();

  auto h_re = Kokkos::create_mirror_view(d_re);
  auto h_im = Kokkos::create_mirror_view(d_im);
  auto h_ref_re = Kokkos::create_mirror_view(d_ref_re);
  auto h_ref_im = Kokkos::create_mirror_view(d_ref_im);
  Kokkos::deep_copy(h_re, d_re);
  Kokkos::deep_copy(h_im, d_im);
  Kokkos::deep_copy(h_ref_re, d_ref_re);
  Kokkos::deep_copy(h_ref_im, d_ref_im);

  // Q(0,0) == Q(1,1) == ca.
  EXPECT_NEAR(h_re(0), h_ref_re(0), kTol);
  EXPECT_NEAR(h_im(0), h_ref_im(0), kTol);
  EXPECT_NEAR(h_re(1), h_ref_re(3), kTol);
  EXPECT_NEAR(h_im(1), h_ref_im(3), kTol);

  // Q(0,1) == -sa*eta*p/rho.
  EXPECT_NEAR(h_re(2), h_ref_re(1), kTol);
  EXPECT_NEAR(h_im(2), h_ref_im(1), kTol);

  // Q(1,0) == sa*rho/(p*eta).
  EXPECT_NEAR(h_re(3), h_ref_re(2), kTol);
  EXPECT_NEAR(h_im(3), h_ref_im(2), kTol);

  // Q(2,2) == 0 (block stays zero).
  EXPECT_NEAR(h_re(4), 0.0, kTol);
  EXPECT_NEAR(h_im(4), 0.0, kTol);
}

// ============================================================================
// Test 4: HalfspaceEigenmatrix
//
// Verify selected entries of E against Tong (2014) A10 analytic expressions.
// ============================================================================

TEST(LayerOperators, HalfspaceEigenmatrix) {
  constexpr double kTol = 1.0e-10;

  // 4 entry pairs: E(0,2), E(1,0), E(0,0), E(2,2).
  Kokkos::View<double *> d_re("re", 4);
  Kokkos::View<double *> d_im("im", 4);
  Kokkos::View<double *> d_ref_re("ref_re", 4);
  Kokkos::View<double *> d_ref_im("ref_im", 4);

  Kokkos::parallel_for("halfspace_eigenmatrix", Kokkos::RangePolicy<>(0, 1),
                       layer_operators_tests_impl::HalfspaceEigenmatrix{
                           d_re, d_im, d_ref_re, d_ref_im });
  Kokkos::fence();

  auto h_re = Kokkos::create_mirror_view(d_re);
  auto h_im = Kokkos::create_mirror_view(d_im);
  auto h_ref_re = Kokkos::create_mirror_view(d_ref_re);
  auto h_ref_im = Kokkos::create_mirror_view(d_ref_im);
  Kokkos::deep_copy(h_re, d_re);
  Kokkos::deep_copy(h_im, d_im);
  Kokkos::deep_copy(h_ref_re, d_ref_re);
  Kokkos::deep_copy(h_ref_im, d_ref_im);

  // E(0,2) == 1.
  EXPECT_NEAR(h_re(0), h_ref_re(2), kTol);
  EXPECT_NEAR(h_im(0), h_ref_im(2), kTol);

  // E(1,0) == 1.
  EXPECT_NEAR(h_re(1), h_ref_re(3), kTol);
  EXPECT_NEAR(h_im(1), h_ref_im(3), kTol);

  // E(0,0) == eta_beta / p.
  EXPECT_NEAR(h_re(2), h_ref_re(0), kTol);
  EXPECT_NEAR(h_im(2), h_ref_im(0), kTol);

  // E(2,2) == 2*rho*vs^2 * eta_alpha / p.
  EXPECT_NEAR(h_re(3), h_ref_re(1), kTol);
  EXPECT_NEAR(h_im(3), h_ref_im(1), kTol);
}

// ============================================================================
// Test 5: CoupleSolidToFluid
//
// Given state {a,b,c,d}: result[0]==b, result[1]==-d, result[2]==c,
// result[3]==-d.
// ============================================================================

TEST(LayerOperators, CoupleSolidToFluid) {
  constexpr double kTol = 1.0e-12;

  Kokkos::View<double *> d_re("re", 4);
  Kokkos::View<double *> d_im("im", 4);

  Kokkos::parallel_for(
      "couple_solid_to_fluid", Kokkos::RangePolicy<>(0, 1),
      layer_operators_tests_impl::CoupleSolidToFluid{ d_re, d_im });
  Kokkos::fence();

  auto h_re = Kokkos::create_mirror_view(d_re);
  auto h_im = Kokkos::create_mirror_view(d_im);
  Kokkos::deep_copy(h_re, d_re);
  Kokkos::deep_copy(h_im, d_im);

  // result[0] == b == (3,4).
  EXPECT_NEAR(h_re(0), 3.0, kTol);
  EXPECT_NEAR(h_im(0), 4.0, kTol);

  // result[1] == -d == -(7,8) == (-7,-8).
  EXPECT_NEAR(h_re(1), -7.0, kTol);
  EXPECT_NEAR(h_im(1), -8.0, kTol);

  // result[2] == c == (5,6).
  EXPECT_NEAR(h_re(2), 5.0, kTol);
  EXPECT_NEAR(h_im(2), 6.0, kTol);

  // result[3] == -d == (-7,-8).
  EXPECT_NEAR(h_re(3), -7.0, kTol);
  EXPECT_NEAR(h_im(3), -8.0, kTol);
}
