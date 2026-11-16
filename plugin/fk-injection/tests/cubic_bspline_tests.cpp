#include "specfem/fk_plugin/utilities/cubic_bspline.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// cubic_bspline_basis: known values at integer and half-integer points
// ---------------------------------------------------------------------------

TEST(CubicBspline, BasisKnownValues) {
  constexpr double kTol = 1.0e-14;

  const double b0 = specfem::fk_plugin::cubic_bspline::cubic_bspline_basis(0.0);
  const double b1 = specfem::fk_plugin::cubic_bspline::cubic_bspline_basis(1.0);
  const double bm1 =
      specfem::fk_plugin::cubic_bspline::cubic_bspline_basis(-1.0);
  const double b2 = specfem::fk_plugin::cubic_bspline::cubic_bspline_basis(2.0);
  const double bm2 =
      specfem::fk_plugin::cubic_bspline::cubic_bspline_basis(-2.0);

  EXPECT_NEAR(b0, 2.0 / 3.0, kTol) << "B3(0) should be 2/3";
  EXPECT_NEAR(b1, 1.0 / 6.0, kTol) << "B3(1) should be 1/6";
  EXPECT_NEAR(bm1, 1.0 / 6.0, kTol) << "B3(-1) should be 1/6";
  EXPECT_NEAR(b2, 0.0, kTol) << "B3(2) should be 0";
  EXPECT_NEAR(bm2, 0.0, kTol) << "B3(-2) should be 0";
  EXPECT_NEAR(bm1 + b0 + b1, 1.0, kTol)
      << "Partition of unity B3(-1)+B3(0)+B3(1) == 1";
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

  std::vector<double> signal(kN);
  for (int i = 0; i < kN; ++i) {
    signal[i] = std::sin(0.3 * static_cast<double>(i)) +
                0.5 * std::cos(0.11 * static_cast<double>(i));
  }

  std::vector<double> coeff(kN);
  specfem::fk_plugin::cubic_bspline::compute_coefficients(signal.data(), kN,
                                                          coeff.data());

  double max_err = 0.0;
  for (int m = 12; m < kN - 12; ++m) {
    double interp = specfem::fk_plugin::cubic_bspline::evaluate(
        coeff.data(), kN, static_cast<double>(m));
    double e = std::fabs(interp - signal[m]);
    if (e > max_err)
      max_err = e;
  }

  EXPECT_LT(max_err, kInterpTol)
      << "Interior interpolation error exceeds " << kInterpTol;
}

// ---------------------------------------------------------------------------
// Exact recovery for a constant signal in the interior
// ---------------------------------------------------------------------------

TEST(CubicBspline, ConstantSignal) {
  constexpr int kN = 32;
  constexpr double kConst = 3.14159;
  constexpr double kConstTol = 1.0e-4;

  std::vector<double> signal(kN, kConst);
  std::vector<double> coeff(kN);
  specfem::fk_plugin::cubic_bspline::compute_coefficients(signal.data(), kN,
                                                          coeff.data());

  double max_err = 0.0;
  for (int m = 8; m < kN - 8; ++m) {
    double interp = specfem::fk_plugin::cubic_bspline::evaluate(
        coeff.data(), kN, static_cast<double>(m));
    double e = std::fabs(interp - kConst);
    if (e > max_err)
      max_err = e;
  }

  EXPECT_LT(max_err, kConstTol)
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
  constexpr double kAbsFloor = 1.0e-10;

  std::vector<double> signal(kN);
  for (int i = 0; i < kN; ++i) {
    signal[i] = std::sin(0.3 * static_cast<double>(i)) +
                0.5 * std::cos(0.11 * static_cast<double>(i));
  }

  std::vector<double> coeff_dbl(kN);
  specfem::fk_plugin::cubic_bspline::compute_coefficients(signal.data(), kN,
                                                          coeff_dbl.data());

  // Narrow-cast to float.
  std::vector<float> coeff_flt(kN);
  for (int i = 0; i < kN; ++i)
    coeff_flt[i] = static_cast<float>(coeff_dbl[i]);

  // Relative errors (with an absolute floor) at non-integer abscissae spread
  // across the interior.
  double max_err = 0.0;
  constexpr double abscissae[5] = { 10.3, 15.7, 20.1, 25.5, 30.9 };
  for (const double t : abscissae) {
    const double val_dbl =
        specfem::fk_plugin::cubic_bspline::evaluate(coeff_dbl.data(), kN, t);
    const double val_flt =
        specfem::fk_plugin::cubic_bspline::evaluate(coeff_flt.data(), kN, t);
    const double denom =
        (std::fabs(val_dbl) > kAbsFloor) ? std::fabs(val_dbl) : kAbsFloor;
    const double rel_err = std::fabs(val_flt - val_dbl) / denom;
    if (rel_err > max_err)
      max_err = rel_err;
  }

  EXPECT_LT(max_err, kRelTol)
      << "evaluate<float> vs evaluate<double> relative error exceeds "
      << kRelTol;
}
