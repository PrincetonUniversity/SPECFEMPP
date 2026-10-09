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
