#include "specfem/utilities/cubic_bspline.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <gtest/gtest.h>

// Device kernels are expressed as named functors rather than extended
// __device__ lambdas: nvcc forbids an extended lambda in gtest's private
// TestBody().  File-unique namespace (not anonymous) keeps the unity build
// ODR-safe.
namespace cubic_bspline_tests_impl {

struct BasisKnownValues {
  Kokkos::View<double *> d_vals;
  KOKKOS_FUNCTION void operator()(const int) const {
    d_vals(0) = specfem::utilities::cubic_bspline::cubic_bspline_basis(0.0);
    d_vals(1) = specfem::utilities::cubic_bspline::cubic_bspline_basis(1.0);
    d_vals(2) = specfem::utilities::cubic_bspline::cubic_bspline_basis(-1.0);
    d_vals(3) = specfem::utilities::cubic_bspline::cubic_bspline_basis(2.0);
    d_vals(4) = specfem::utilities::cubic_bspline::cubic_bspline_basis(-2.0);
    // Partition of unity: B3(-1) + B3(0) + B3(1) = 1
    d_vals(5) = (specfem::utilities::cubic_bspline::cubic_bspline_basis(-1.0) +
                 specfem::utilities::cubic_bspline::cubic_bspline_basis(0.0) +
                 specfem::utilities::cubic_bspline::cubic_bspline_basis(1.0));
  }
};

struct InteriorRoundTrip {
  Kokkos::View<double *> d_signal;
  Kokkos::View<double *> d_coeff;
  Kokkos::View<double *> d_err;
  KOKKOS_FUNCTION void operator()(const int) const {
    const int kN = static_cast<int>(d_signal.extent(0));
    specfem::utilities::cubic_bspline::compute_coefficients(d_signal.data(), kN,
                                                            d_coeff.data());

    double max_err = 0.0;
    for (int m = 12; m < kN - 12; ++m) {
      double interp = specfem::utilities::cubic_bspline::evaluate(
          d_coeff.data(), kN, static_cast<double>(m));
      double e = Kokkos::fabs(interp - d_signal(m));
      if (e > max_err)
        max_err = e;
    }
    d_err(0) = max_err;
  }
};

struct ConstantSignal {
  Kokkos::View<double *> d_signal;
  Kokkos::View<double *> d_coeff;
  Kokkos::View<double *> d_err;
  double const_val;
  KOKKOS_FUNCTION void operator()(const int) const {
    const int kN = static_cast<int>(d_signal.extent(0));
    specfem::utilities::cubic_bspline::compute_coefficients(d_signal.data(), kN,
                                                            d_coeff.data());
    double max_err = 0.0;
    for (int m = 8; m < kN - 8; ++m) {
      double interp = specfem::utilities::cubic_bspline::evaluate(
          d_coeff.data(), kN, static_cast<double>(m));
      double e = Kokkos::fabs(interp - const_val);
      if (e > max_err)
        max_err = e;
    }
    d_err(0) = max_err;
  }
};

} // namespace cubic_bspline_tests_impl

// ---------------------------------------------------------------------------
// EvaluateFloatCoefficientsMatchesDouble helpers
// ---------------------------------------------------------------------------
namespace cubic_bspline_tests_impl {

// Compare evaluate<float> against evaluate<double> for several abscissae.
// Accumulates relative errors (with an absolute floor) into d_max_err.
struct FloatVsDouble {
  Kokkos::View<double *> d_coeff_dbl;
  Kokkos::View<float *> d_coeff_flt;
  Kokkos::View<double *> d_max_err;
  int n;
  KOKKOS_FUNCTION void operator()(const int) const {
    constexpr double abs_floor = 1.0e-10;
    double max_err = 0.0;

    // Non-integer abscissae spread across the interior.
    constexpr double abscissae[5] = { 10.3, 15.7, 20.1, 25.5, 30.9 };
    for (int a = 0; a < 5; ++a) {
      const double t = abscissae[a];
      const double val_dbl =
          specfem::utilities::cubic_bspline::evaluate(d_coeff_dbl.data(), n, t);
      const double val_flt =
          specfem::utilities::cubic_bspline::evaluate(d_coeff_flt.data(), n, t);
      const double denom = (Kokkos::fabs(val_dbl) > abs_floor)
                               ? Kokkos::fabs(val_dbl)
                               : abs_floor;
      const double rel_err = Kokkos::fabs(val_flt - val_dbl) / denom;
      if (rel_err > max_err)
        max_err = rel_err;
    }
    d_max_err(0) = max_err;
  }
};

} // namespace cubic_bspline_tests_impl

// ---------------------------------------------------------------------------
// cubic_bspline_basis: known values at integer and half-integer points
// ---------------------------------------------------------------------------

TEST(CubicBspline, BasisKnownValues) {
  constexpr double kTol = 1.0e-14;
  Kokkos::View<double *> d_vals("bspline_vals", 6);

  Kokkos::parallel_for("bspline_basis", Kokkos::RangePolicy<>(0, 1),
                       cubic_bspline_tests_impl::BasisKnownValues{ d_vals });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(d_vals);
  Kokkos::deep_copy(h, d_vals);

  EXPECT_NEAR(h(0), 2.0 / 3.0, kTol) << "B3(0) should be 2/3";
  EXPECT_NEAR(h(1), 1.0 / 6.0, kTol) << "B3(1) should be 1/6";
  EXPECT_NEAR(h(2), 1.0 / 6.0, kTol) << "B3(-1) should be 1/6";
  EXPECT_NEAR(h(3), 0.0, kTol) << "B3(2) should be 0";
  EXPECT_NEAR(h(4), 0.0, kTol) << "B3(-2) should be 0";
  EXPECT_NEAR(h(5), 1.0, kTol) << "Partition of unity B3(-1)+B3(0)+B3(1) == 1";
}

// ---------------------------------------------------------------------------
// Round-trip interpolation in the INTERIOR of a smooth signal
//
// The reference prefilter uses a specific edge initial condition (the Fortran
// causal IIR has a non-standard boundary treatment), so evaluate() is only
// guaranteed to match signal[m] for INTERIOR indices far from the ends.
// We test m in [12, n-12] where n=64.
// ---------------------------------------------------------------------------

TEST(CubicBspline, InteriorRoundTrip) {
  constexpr int kN = 64;
  constexpr double kInterpTol = 1.0e-4;

  // Build signal on host.
  Kokkos::View<double *> d_signal("signal", kN);
  Kokkos::View<double *> d_coeff("coeff", kN);
  Kokkos::View<double *> d_err("err", 1);

  {
    auto h_sig = Kokkos::create_mirror_view(d_signal);
    for (int i = 0; i < kN; ++i) {
      h_sig(i) = Kokkos::sin(0.3 * static_cast<double>(i)) +
                 0.5 * Kokkos::cos(0.11 * static_cast<double>(i));
    }
    Kokkos::deep_copy(d_signal, h_sig);
  }

  Kokkos::parallel_for(
      "bspline_roundtrip", Kokkos::RangePolicy<>(0, 1),
      cubic_bspline_tests_impl::InteriorRoundTrip{ d_signal, d_coeff, d_err });
  Kokkos::fence();

  auto h_err = Kokkos::create_mirror_view(d_err);
  Kokkos::deep_copy(h_err, d_err);

  EXPECT_LT(h_err(0), kInterpTol)
      << "Interior interpolation error exceeds " << kInterpTol;
}

// ---------------------------------------------------------------------------
// Exact recovery for a constant signal in the interior
// ---------------------------------------------------------------------------

TEST(CubicBspline, ConstantSignal) {
  constexpr int kN = 32;
  constexpr double kConst = 3.14159;
  constexpr double kConstTol = 1.0e-4;

  Kokkos::View<double *> d_signal("const_sig", kN);
  Kokkos::View<double *> d_coeff("const_coeff", kN);
  Kokkos::View<double *> d_err("const_err", 1);

  {
    auto h_sig = Kokkos::create_mirror_view(d_signal);
    for (int i = 0; i < kN; ++i)
      h_sig(i) = kConst;
    Kokkos::deep_copy(d_signal, h_sig);
  }

  double const_val = kConst;
  Kokkos::parallel_for("bspline_const", Kokkos::RangePolicy<>(0, 1),
                       cubic_bspline_tests_impl::ConstantSignal{
                           d_signal, d_coeff, d_err, const_val });
  Kokkos::fence();

  auto h_err = Kokkos::create_mirror_view(d_err);
  Kokkos::deep_copy(h_err, d_err);

  EXPECT_LT(h_err(0), kConstTol)
      << "Constant signal should be recovered in interior to within "
      << kConstTol;
}

// ---------------------------------------------------------------------------
// evaluate<float> matches evaluate<double> to a relative tolerance of 1e-6
//
// float coefficients introduce ~1e-7 relative rounding vs. double; 1e-6
// gives a safe margin without hiding actual template bugs.
// ---------------------------------------------------------------------------

TEST(CubicBspline, EvaluateFloatCoefficientsMatchesDouble) {
  constexpr int kN = 64;
  constexpr double kRelTol = 1.0e-6;

  // Build signal on host and compute double coefficients.
  std::vector<double> signal_h(kN);
  for (int i = 0; i < kN; ++i) {
    signal_h[i] = Kokkos::sin(0.3 * static_cast<double>(i)) +
                  0.5 * Kokkos::cos(0.11 * static_cast<double>(i));
  }

  std::vector<double> coeff_dbl_h(kN);
  specfem::utilities::cubic_bspline::compute_coefficients(signal_h.data(), kN,
                                                          coeff_dbl_h.data());

  // Narrow-cast to float.
  std::vector<float> coeff_flt_h(kN);
  for (int i = 0; i < kN; ++i)
    coeff_flt_h[i] = static_cast<float>(coeff_dbl_h[i]);

  // Upload both coefficient arrays to device.
  Kokkos::View<double *> d_coeff_dbl("coeff_dbl", kN);
  Kokkos::View<float *> d_coeff_flt("coeff_flt", kN);
  Kokkos::View<double *> d_max_err("flt_max_err", 1);

  {
    auto h_dbl = Kokkos::create_mirror_view(d_coeff_dbl);
    auto h_flt = Kokkos::create_mirror_view(d_coeff_flt);
    for (int i = 0; i < kN; ++i) {
      h_dbl(i) = coeff_dbl_h[i];
      h_flt(i) = coeff_flt_h[i];
    }
    Kokkos::deep_copy(d_coeff_dbl, h_dbl);
    Kokkos::deep_copy(d_coeff_flt, h_flt);
  }

  Kokkos::parallel_for("bspline_float_vs_double", Kokkos::RangePolicy<>(0, 1),
                       cubic_bspline_tests_impl::FloatVsDouble{
                           d_coeff_dbl, d_coeff_flt, d_max_err, kN });
  Kokkos::fence();

  auto h_max_err = Kokkos::create_mirror_view(d_max_err);
  Kokkos::deep_copy(h_max_err, d_max_err);

  EXPECT_LT(h_max_err(0), kRelTol)
      << "evaluate<float> vs evaluate<double> relative error exceeds "
      << kRelTol;
}
