#include "specfem/utilities/fft.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <gtest/gtest.h>

#include <vector>

// ---------------------------------------------------------------------------
// Normalization note (documented from analytical derivation)
//
// Both transform() (zign=-1) and inverse() apply the same direction DFT
// (e^{+2pi*i*n*k/lx} exponent) with scale factor 1/(lx*dt).
//
// For a real signal r[n] with no energy at the Nyquist bin (k = lx/2):
//   inverse( forward(r) )[n]  =  r[n] / lx   (with dt=1)
//
// restructure_spectrum is a no-op when the spectrum already satisfies
// Hermitian symmetry and zero Nyquist — which is the case for the forward
// transform of a band-limited real signal.  We use a cosine at k0=1 for
// the round-trip test to ensure this property holds exactly.
// ---------------------------------------------------------------------------

// Device kernels are expressed as named functors rather than extended
// __device__ lambdas: nvcc forbids an extended lambda in gtest's private
// TestBody().  File-unique namespace (not anonymous) keeps the unity build
// ODR-safe.
namespace fft_tests_impl {

struct RoundTrip {
  Kokkos::View<Kokkos::complex<double> *> d_buf;
  Kokkos::View<double *> d_r;
  Kokkos::View<double *> d_out;
  Kokkos::View<int *> bit_reversal_view;
  Kokkos::View<Kokkos::complex<double> *> twiddles_view;
  int power;
  double dt;
  KOKKOS_FUNCTION void operator()(const int) const {
    const int kLx = static_cast<int>(d_r.extent(0));
    // Load real signal into complex buffer.
    for (int n = 0; n < kLx; ++n) {
      d_buf(n) = Kokkos::complex<double>(d_r(n), 0.0);
    }

    // Forward transform (zign=-1, dt=1).
    specfem::utilities::fft::transform(power, d_buf.data(), -1.0, dt,
                                       bit_reversal_view.data(),
                                       twiddles_view.data());

    // Inverse: restructure_spectrum + transform again.
    // inverse() overwrites d_buf and writes real parts to d_out.
    specfem::utilities::fft::inverse(power, d_buf.data(), -1.0, dt,
                                     d_out.data(), bit_reversal_view.data(),
                                     twiddles_view.data());
  }
};

struct CosineBin {
  Kokkos::View<Kokkos::complex<double> *> d_buf;
  Kokkos::View<double *> d_r;
  Kokkos::View<int *> bit_reversal_view;
  Kokkos::View<Kokkos::complex<double> *> twiddles_view;
  int power;
  double dt;
  KOKKOS_FUNCTION void operator()(const int) const {
    const int kLx = static_cast<int>(d_r.extent(0));
    for (int n = 0; n < kLx; ++n) {
      d_buf(n) = Kokkos::complex<double>(d_r(n), 0.0);
    }
    specfem::utilities::fft::transform(power, d_buf.data(), -1.0, dt,
                                       bit_reversal_view.data(),
                                       twiddles_view.data());
  }
};

struct Restructure {
  Kokkos::View<Kokkos::complex<double> *> d_s;
  Kokkos::View<double *> d_checks;
  KOKKOS_FUNCTION void operator()(const int) const {
    const int lx = static_cast<int>(d_s.extent(0));
    const int nhalf = lx / 2;

    // Set first half with non-symmetric values.
    for (int k = 0; k < nhalf; ++k) {
      d_s(k) = Kokkos::complex<double>(static_cast<double>(k + 1),
                                       static_cast<double>(k));
    }
    // Leave second half as zero before restructure.
    for (int k = nhalf; k < lx; ++k) {
      d_s(k) = Kokkos::complex<double>(0.0, 0.0);
    }

    specfem::utilities::fft::restructure_spectrum(d_s.data(), nhalf);

    // Check 1: s[nhalf] == 0 (Nyquist zeroed).
    d_checks(0) = d_s(nhalf).real();
    d_checks(1) = d_s(nhalf).imag();

    // Check 2: s[0] is real (zero imaginary).
    d_checks(2) = d_s(0).imag();

    // Check 3: s[nhalf+i-1] == conj(s[nhalf+1-i]) for i=2 and i=3.
    // i=2: s[nhalf+1] == conj(s[nhalf-1])
    double diff_r2 = d_s(nhalf + 1).real() - d_s(nhalf - 1).real();
    double diff_i2 =
        d_s(nhalf + 1).imag() + d_s(nhalf - 1).imag(); // conj flips sign
    d_checks(3) = diff_r2;
    d_checks(4) = diff_i2;

    // i=3: s[nhalf+2] == conj(s[nhalf-2])
    double diff_r3 = d_s(nhalf + 2).real() - d_s(nhalf - 2).real();
    double diff_i3 = d_s(nhalf + 2).imag() + d_s(nhalf - 2).imag();
    d_checks(5) = Kokkos::fabs(diff_r3) + Kokkos::fabs(diff_i3);
  }
};

} // namespace fft_tests_impl

// ---------------------------------------------------------------------------
// Round-trip: forward → inverse recovers r[n] / lx  (within kTol)
// ---------------------------------------------------------------------------

TEST(Fft, RoundTripCosine) {
  constexpr int kPower = 5; // lx = 32
  constexpr int kLx = 1 << kPower;
  constexpr double kDt = 1.0;
  constexpr double kTol = 1.0e-9;
  // Build tables on host, copy to device (make_fft_tables is host-callable).
  specfem::utilities::fft::FftTables tables =
      specfem::utilities::fft::make_fft_tables(kPower, -1.0);

  // Build the real signal r[n] = cos(2π*n/lx) on host.
  std::vector<double> r_host(kLx);
  constexpr double two_pi = 2.0 * 3.141592653589793;
  for (int n = 0; n < kLx; ++n) {
    r_host[n] =
        Kokkos::cos(two_pi * static_cast<double>(n) / static_cast<double>(kLx));
  }

  // Device scratch: complex input buffer of length lx.
  Kokkos::View<Kokkos::complex<double> *> d_buf("fft_buf", kLx);
  Kokkos::View<double *> d_out("fft_out", kLx);
  Kokkos::View<double *> d_r("signal", kLx);

  // Copy signal to device view.
  {
    auto h_r = Kokkos::create_mirror_view(d_r);
    for (int n = 0; n < kLx; ++n)
      h_r(n) = r_host[n];
    Kokkos::deep_copy(d_r, h_r);
  }

  Kokkos::parallel_for(
      "fft_roundtrip", Kokkos::RangePolicy<>(0, 1),
      fft_tests_impl::RoundTrip{ d_buf, d_r, d_out, tables.bit_reversal,
                                 tables.twiddles, tables.power, kDt });
  Kokkos::fence();

  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);

  // Expected: out[n] = r[n] / lx  (analytical result, see normalization note).
  for (int n = 0; n < kLx; ++n) {
    EXPECT_NEAR(h_out(n), r_host[n] / static_cast<double>(kLx), kTol)
        << "mismatch at n=" << n;
  }
}

// ---------------------------------------------------------------------------
// Pure cosine energy concentrated at bins k0 and lx-k0
// ---------------------------------------------------------------------------

TEST(Fft, CosineBinConcentration) {
  constexpr int kPower = 5; // lx = 32
  constexpr int kLx = 1 << kPower;
  constexpr double kDt = 1.0;
  specfem::utilities::fft::FftTables tables =
      specfem::utilities::fft::make_fft_tables(kPower, -1.0);

  Kokkos::View<Kokkos::complex<double> *> d_buf("fft_cos", kLx);
  Kokkos::View<double *> d_r("signal_cos", kLx);

  constexpr int k0 = 3; // interior bin, away from DC and Nyquist
  {
    auto h_r = Kokkos::create_mirror_view(d_r);
    constexpr double two_pi = 2.0 * 3.141592653589793;
    for (int n = 0; n < kLx; ++n) {
      h_r(n) = Kokkos::cos(two_pi * static_cast<double>(k0) *
                           static_cast<double>(n) / static_cast<double>(kLx));
    }
    Kokkos::deep_copy(d_r, h_r);
  }

  Kokkos::parallel_for(
      "fft_cosbin", Kokkos::RangePolicy<>(0, 1),
      fft_tests_impl::CosineBin{ d_buf, d_r, tables.bit_reversal,
                                 tables.twiddles, tables.power, kDt });
  Kokkos::fence();

  auto h_buf = Kokkos::create_mirror_view(d_buf);
  Kokkos::deep_copy(h_buf, d_buf);

  // Compute magnitudes.
  std::vector<double> mag(kLx);
  for (int k = 0; k < kLx; ++k) {
    mag[k] = Kokkos::sqrt(h_buf(k).real() * h_buf(k).real() +
                          h_buf(k).imag() * h_buf(k).imag());
  }

  // Total energy.
  double total = 0.0;
  for (int k = 0; k < kLx; ++k)
    total += mag[k];

  // The two signal bins should hold essentially all the energy.
  double signal_energy = mag[k0] + mag[kLx - k0];
  EXPECT_GT(signal_energy / total, 0.99)
      << "Expected energy concentrated at bins " << k0 << " and " << kLx - k0;

  // All other bins should be near zero.
  for (int k = 0; k < kLx; ++k) {
    if (k == k0 || k == kLx - k0)
      continue;
    EXPECT_LT(mag[k], 1.0e-10) << "unexpected energy at bin k=" << k;
  }
}

// ---------------------------------------------------------------------------
// restructure_spectrum: Hermitian symmetry and zero imaginary at DC / Nyquist
// ---------------------------------------------------------------------------

TEST(Fft, RestructureSpectrum) {
  // Fill a half-spectrum with arbitrary values then restructure.
  constexpr int nhalf = 8;
  constexpr int lx = 2 * nhalf;

  Kokkos::View<Kokkos::complex<double> *> d_s("rspec", lx);
  Kokkos::View<double *> d_checks("checks", 6); // results to assert on host

  Kokkos::parallel_for("rspec_test", Kokkos::RangePolicy<>(0, 1),
                       fft_tests_impl::Restructure{ d_s, d_checks });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(d_checks);
  Kokkos::deep_copy(h, d_checks);

  EXPECT_NEAR(h(0), 0.0, 1.0e-14) << "s[nhalf].real should be 0";
  EXPECT_NEAR(h(1), 0.0, 1.0e-14) << "s[nhalf].imag should be 0";
  EXPECT_NEAR(h(2), 0.0, 1.0e-14) << "s[0].imag should be 0 (forced real)";
  EXPECT_NEAR(h(3), 0.0, 1.0e-14)
      << "Hermitian real: s[nhalf+1].re == s[nhalf-1].re";
  EXPECT_NEAR(h(4), 0.0, 1.0e-14)
      << "Hermitian imag: s[nhalf+1].im == -s[nhalf-1].im";
  EXPECT_NEAR(h(5), 0.0, 1.0e-14)
      << "Hermitian: s[nhalf+2] == conj(s[nhalf-2])";
}
