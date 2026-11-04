#include "fk_injection/fk_result.hpp"
#include "fk_injection/utilities/cubic_bspline.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers live in a named namespace (unity-build safe, no anonymous namespace).
// ---------------------------------------------------------------------------
namespace fk_result_tests_impl {

// Row-major (point, component, step) index into the output of sample().
inline int out_index(int point, int component, int step, int number_of_steps) {
  return (point * 7 + component) * number_of_steps + step;
}

// Build a synthetic FkResult with 1 group, 5 series, ncoef coefficients.
// Series 0 (S0): linear ramp — coefficient[k] = k (reproduce ramp in interior).
// Series 1 (S1): constant 2.0
// Series 2 (S2): constant 3.0
// Series 3 (S3): constant 5.0
// Series 4 (S4): constant 7.0
//
// 2 elastic points (group 0), offsets 10.0 and 10.5.
// cos_phi=0.6  sin_phi=0.8  resampling_rate=1  has_traction=true
// point_material: xi1=2.0 xim=0.5 bulk=0.25 nx=0 ny=0 nz=1
static fk_injection::FkResult make_synthetic_result(int ncoef = 64,
                                                    int resampling_rate = 1) {
  const int ngroups = 1;
  const int nseries = 5;
  const int npoints = 2;

  // --- coefficient array, row-major (group, series, coefficient) ---
  std::vector<fk_injection::type_real> coefficients(ngroups * nseries * ncoef);

  // S0: pre-filter a linear ramp so evaluate() reproduces the ramp in the
  // interior.
  {
    std::vector<double> sig(ncoef), coef(ncoef);
    for (int k = 0; k < ncoef; ++k) {
      sig[k] = static_cast<double>(k);
    }
    fk_injection::cubic_bspline::compute_coefficients(sig.data(), ncoef,
                                                      coef.data());
    for (int k = 0; k < ncoef; ++k) {
      coefficients[k] = static_cast<fk_injection::type_real>(coef[k]);
    }
  }

  // S1..S4: constants (pre-filter of a constant is the same constant).
  const fk_injection::type_real s1_val =
      static_cast<fk_injection::type_real>(2.0);
  const fk_injection::type_real s2_val =
      static_cast<fk_injection::type_real>(3.0);
  const fk_injection::type_real s3_val =
      static_cast<fk_injection::type_real>(5.0);
  const fk_injection::type_real s4_val =
      static_cast<fk_injection::type_real>(7.0);
  {
    std::vector<double> sig(ncoef), coef_d(ncoef);
    auto fill_constant = [&](int s_idx, double val) {
      for (int k = 0; k < ncoef; ++k) {
        sig[k] = val;
      }
      fk_injection::cubic_bspline::compute_coefficients(sig.data(), ncoef,
                                                        coef_d.data());
      for (int k = 0; k < ncoef; ++k) {
        coefficients[s_idx * ncoef + k] =
            static_cast<fk_injection::type_real>(coef_d[k]);
      }
    };
    fill_constant(1, static_cast<double>(s1_val));
    fill_constant(2, static_cast<double>(s2_val));
    fill_constant(3, static_cast<double>(s3_val));
    fill_constant(4, static_cast<double>(s4_val));
  }

  // --- point_group ---
  const std::vector<int> point_group(npoints, 0);

  // --- point_offset ---
  const std::vector<double> point_offset = { 10.0, 10.5 };

  // --- point_material: [xi1, xim, bulk, nx, ny, nz] ---
  std::vector<fk_injection::type_real> point_material(npoints * 6);
  for (int ip = 0; ip < npoints; ++ip) {
    point_material[ip * 6 + 0] =
        static_cast<fk_injection::type_real>(2.0); // xi1
    point_material[ip * 6 + 1] =
        static_cast<fk_injection::type_real>(0.5); // xim
    point_material[ip * 6 + 2] =
        static_cast<fk_injection::type_real>(0.25); // bulk
    point_material[ip * 6 + 3] =
        static_cast<fk_injection::type_real>(0.0); // nx
    point_material[ip * 6 + 4] =
        static_cast<fk_injection::type_real>(0.0); // ny
    point_material[ip * 6 + 5] =
        static_cast<fk_injection::type_real>(1.0); // nz
  }

  // --- point_is_acoustic ---
  const std::vector<int> point_is_acoustic(npoints, 0);

  const fk_injection::type_real cos_phi =
      static_cast<fk_injection::type_real>(0.6);
  const fk_injection::type_real sin_phi =
      static_cast<fk_injection::type_real>(0.8);

  return fk_injection::FkResult(coefficients, ngroups, ncoef, point_group,
                                point_offset, point_material, point_is_acoustic,
                                cos_phi, sin_phi, resampling_rate,
                                /*has_traction=*/true);
}

} // namespace fk_result_tests_impl

// ---------------------------------------------------------------------------
// Test suite: FkResult
// ---------------------------------------------------------------------------

// 1. SampleOffsetAndAzimuth
//    comp0 = cos_phi * S0(a), comp1 = sin_phi * S0(a), comp2 = S1(a) = 2.0
//    where a = k/1 + offset(point).  Checked for interior steps k in [2, 17].
TEST(FkResult, SampleOffsetAndAzimuth) {
  const int ncoef = 64;
  const int N = 20;
  const auto result = fk_result_tests_impl::make_synthetic_result(ncoef, 1);

  std::vector<fk_injection::type_real> out(2 * 7 * N);
  result.sample(0, N, 7, out.data());

  const double cos_phi = 0.6;
  const double sin_phi = 0.8;
  const double offsets[2] = { 10.0, 10.5 };

  // Avoid spline boundary: k in [2, N-3]; a = k + offset stays in [12, 27.5],
  // well within [2, ncoef-3]=[2,61].
  for (int p = 0; p < 2; ++p) {
    for (int k = 2; k <= N - 3; ++k) {
      const double a = static_cast<double>(k) + offsets[p];
      const double c0 =
          static_cast<double>(out[fk_result_tests_impl::out_index(p, 0, k, N)]);
      const double c1 =
          static_cast<double>(out[fk_result_tests_impl::out_index(p, 1, k, N)]);
      const double c2 =
          static_cast<double>(out[fk_result_tests_impl::out_index(p, 2, k, N)]);
      // comp0 = cos_phi * S0(a) ≈ cos_phi * a  (spline reproduces the ramp)
      EXPECT_NEAR(c0, cos_phi * a, 1.0e-3 * a)
          << "comp0 failed at point=" << p << " k=" << k;
      // comp1 = sin_phi * S0(a)
      EXPECT_NEAR(c1, sin_phi * a, 1.0e-3 * a)
          << "comp1 failed at point=" << p << " k=" << k;
      // comp2 = S1(a) = constant 2.0
      EXPECT_NEAR(c2, 2.0, 1.0e-3)
          << "comp2 failed at point=" << p << " k=" << k;
    }
  }
}

// 2. SampleTractionWeights
//    With S2=3, S3=5, S4=7, xi1=2, xim=0.5, bulk=0.25, n=(0,0,1):
//    sigma_rr = 2*3 - 4*0.5*7 = -8
//    sigma_rz = 5
//    sigma_zz = 3
//    sigma_tt = 0.25*(-8+3) = -1.25
//    With n=(0,0,1): tx=sigma_rz*cos_phi=3.0, ty=sigma_rz*sin_phi=4.0, tz=3.0
//    comp3=tx, comp4=ty, comp5=tz; comp6=0 (elastic)
TEST(FkResult, SampleTractionWeights) {
  const int ncoef = 64;
  const int N = 20;
  const auto result = fk_result_tests_impl::make_synthetic_result(ncoef, 1);

  std::vector<fk_injection::type_real> out(2 * 7 * N);
  result.sample(0, N, 7, out.data());

  // Constants in S2..S4 → same value at any interior step.
  const int k_check = 10;
  for (int p = 0; p < 2; ++p) {
    EXPECT_NEAR(static_cast<double>(
                    out[fk_result_tests_impl::out_index(p, 3, k_check, N)]),
                3.0, 1.0e-3)
        << "comp3 (tx) failed at point=" << p;
    EXPECT_NEAR(static_cast<double>(
                    out[fk_result_tests_impl::out_index(p, 4, k_check, N)]),
                4.0, 1.0e-3)
        << "comp4 (ty) failed at point=" << p;
    EXPECT_NEAR(static_cast<double>(
                    out[fk_result_tests_impl::out_index(p, 5, k_check, N)]),
                3.0, 1.0e-3)
        << "comp5 (tz) failed at point=" << p;
    // comp6: pressure = 0 for elastic points
    EXPECT_NEAR(static_cast<double>(
                    out[fk_result_tests_impl::out_index(p, 6, k_check, N)]),
                0.0, 1.0e-6)
        << "comp6 (pressure) should be 0 for elastic point=" << p;
  }
}

// 3. SampleNpMapping
//    With resampling_rate=2, the abscissa for SEM step k is k/2 + offset.
//    Verify comp0 = cos_phi * (k/2 + offset) for interior k.
TEST(FkResult, SampleNpMapping) {
  const int ncoef = 64;
  const int N = 20;
  // resampling_rate = 2; offsets are 10.0 and 10.5
  const auto result = fk_result_tests_impl::make_synthetic_result(ncoef, 2);

  std::vector<fk_injection::type_real> out(2 * 7 * N);
  result.sample(0, N, 7, out.data());

  const double cos_phi = 0.6;
  const double offsets[2] = { 10.0, 10.5 };

  // a = k/2 + offset; for k in [2, N-3] a ranges from [11.0, 20.0], safely
  // within [2, 61].
  for (int p = 0; p < 2; ++p) {
    for (int k = 2; k <= N - 3; ++k) {
      const double a = static_cast<double>(k) / 2.0 + offsets[p];
      const double c0 =
          static_cast<double>(out[fk_result_tests_impl::out_index(p, 0, k, N)]);
      EXPECT_NEAR(c0, cos_phi * a, 1.0e-3 * a)
          << "NP mapping failed at point=" << p << " k=" << k;
    }
  }
}
