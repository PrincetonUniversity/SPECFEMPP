#include "specfem/fk_plugin/utilities/fft.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Normalization note (documented from analytical derivation)
//
// zign = +1 is the forward transform (kernel e^{-2pi*i*n*k/lx}, scale dt);
// zign = -1 is the inverse (kernel e^{+2pi*i*n*k/lx}, scale 1/(lx*dt)).
//
// RoundTripCosine applies the zign = -1 transform twice (transform() then
// inverse()).  For a real signal r[n] with no energy at the Nyquist bin
// (k = lx/2) this gives the time-reversed signal r[-n] / lx (with dt=1);
// a cosine is time-reversal invariant, so the expected output is r[n] / lx.
// RoundTripAsymmetric covers the proper forward (+1) / inverse (-1) pair on
// a signal that is not time-reversal invariant.
//
// restructure_spectrum is a no-op when the spectrum already satisfies
// Hermitian symmetry and zero Nyquist — which is the case for the
// transform of a band-limited real signal.
// ---------------------------------------------------------------------------

namespace fft_tests_impl {

// Real signal as a complex buffer.
std::vector<std::complex<double>> to_complex(const std::vector<double> &r) {
  std::vector<std::complex<double>> buf(r.size());
  for (std::size_t n = 0; n < r.size(); ++n) {
    buf[n] = std::complex<double>(r[n], 0.0);
  }
  return buf;
}

} // namespace fft_tests_impl

// ---------------------------------------------------------------------------
// Round-trip: forward → inverse recovers r[n] / lx  (within kTol)
// ---------------------------------------------------------------------------

TEST(Fft, RoundTripCosine) {
  constexpr int kPower = 5; // lx = 32
  constexpr int kLx = 1 << kPower;
  constexpr double kDt = 1.0;
  constexpr double kTol = 1.0e-9;
  const auto tables = specfem::fk_plugin::fft::make_fft_tables(kPower, -1.0);

  // Build the real signal r[n] = cos(2π*n/lx).
  std::vector<double> r(kLx);
  constexpr double two_pi = 2.0 * 3.141592653589793;
  for (int n = 0; n < kLx; ++n) {
    r[n] = std::cos(two_pi * static_cast<double>(n) / static_cast<double>(kLx));
  }

  auto buf = fft_tests_impl::to_complex(r);
  std::vector<double> out(kLx);

  // zign=-1 transform (inverse kernel), dt=1.
  specfem::fk_plugin::fft::transform(kPower, buf.data(), -1.0, kDt,
                                     tables.bit_reversal.data(),
                                     tables.twiddles.data());

  // Inverse: restructure_spectrum + transform again.
  // inverse() overwrites buf and writes real parts to out.
  specfem::fk_plugin::fft::inverse(kPower, buf.data(), -1.0, kDt, out.data(),
                                   tables.bit_reversal.data(),
                                   tables.twiddles.data());

  // Expected: out[n] = r[n] / lx  (analytical result, see normalization note).
  for (int n = 0; n < kLx; ++n) {
    EXPECT_NEAR(out[n], r[n] / static_cast<double>(kLx), kTol)
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
  const auto tables = specfem::fk_plugin::fft::make_fft_tables(kPower, -1.0);

  constexpr int k0 = 3; // interior bin, away from DC and Nyquist
  std::vector<double> r(kLx);
  constexpr double two_pi = 2.0 * 3.141592653589793;
  for (int n = 0; n < kLx; ++n) {
    r[n] = std::cos(two_pi * static_cast<double>(k0) * static_cast<double>(n) /
                    static_cast<double>(kLx));
  }

  auto buf = fft_tests_impl::to_complex(r);
  specfem::fk_plugin::fft::transform(kPower, buf.data(), -1.0, kDt,
                                     tables.bit_reversal.data(),
                                     tables.twiddles.data());

  // Compute magnitudes.
  std::vector<double> mag(kLx);
  for (int k = 0; k < kLx; ++k) {
    mag[k] = std::abs(buf[k]);
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

  // First half non-symmetric; second half zero before restructure.
  std::vector<std::complex<double>> s(lx);
  for (int k = 0; k < nhalf; ++k) {
    s[k] = std::complex<double>(static_cast<double>(k + 1),
                                static_cast<double>(k));
  }

  specfem::fk_plugin::fft::restructure_spectrum(s.data(), nhalf);

  EXPECT_NEAR(s[nhalf].real(), 0.0, 1.0e-14) << "s[nhalf].real should be 0";
  EXPECT_NEAR(s[nhalf].imag(), 0.0, 1.0e-14) << "s[nhalf].imag should be 0";
  EXPECT_NEAR(s[0].imag(), 0.0, 1.0e-14)
      << "s[0].imag should be 0 (forced real)";
  EXPECT_NEAR(s[nhalf + 1].real() - s[nhalf - 1].real(), 0.0, 1.0e-14)
      << "Hermitian real: s[nhalf+1].re == s[nhalf-1].re";
  EXPECT_NEAR(s[nhalf + 1].imag() + s[nhalf - 1].imag(), 0.0, 1.0e-14)
      << "Hermitian imag: s[nhalf+1].im == -s[nhalf-1].im";
  EXPECT_NEAR(std::abs(s[nhalf + 2] - std::conj(s[nhalf - 2])), 0.0, 1.0e-14)
      << "Hermitian: s[nhalf+2] == conj(s[nhalf-2])";
}

// ---------------------------------------------------------------------------
// Forward (+1) → inverse (-1) on an asymmetric signal recovers r[n] exactly
// ---------------------------------------------------------------------------

TEST(Fft, RoundTripAsymmetric) {
  constexpr int kPower = 6; // lx = 64
  constexpr int kLx = 1 << kPower;
  constexpr double kDt = 0.01;
  constexpr double kTol = 1.0e-10;
  constexpr double two_pi = 2.0 * 3.141592653589793;

  const auto fwd = specfem::fk_plugin::fft::make_fft_tables(kPower, 1.0);
  const auto inv = specfem::fk_plugin::fft::make_fft_tables(kPower, -1.0);

  // Real, band-limited (no Nyquist energy), not time-reversal invariant.
  std::vector<double> r(kLx);
  for (int n = 0; n < kLx; ++n) {
    const double x = two_pi * static_cast<double>(n) / kLx;
    r[n] = 0.3 + std::cos(x) + 0.5 * std::sin(3.0 * x + 0.7) +
           0.2 * std::sin(5.0 * x);
  }

  auto buf = fft_tests_impl::to_complex(r);
  std::vector<double> out(kLx);
  specfem::fk_plugin::fft::transform(kPower, buf.data(), 1.0, kDt,
                                     fwd.bit_reversal.data(),
                                     fwd.twiddles.data());
  specfem::fk_plugin::fft::inverse(kPower, buf.data(), -1.0, kDt, out.data(),
                                   inv.bit_reversal.data(),
                                   inv.twiddles.data());

  for (int n = 0; n < kLx; ++n) {
    EXPECT_NEAR(out[n], r[n], kTol) << "mismatch at n=" << n;
  }
}

// ---------------------------------------------------------------------------
// Forward kernel sign: sin(2π k0 n / lx) → X[k0] = -i lx/2, X[lx-k0] = +i lx/2
// ---------------------------------------------------------------------------

TEST(Fft, ForwardSineBinSign) {
  constexpr int kPower = 5; // lx = 32
  constexpr int kLx = 1 << kPower;
  constexpr int k0 = 3;
  constexpr double kTol = 1.0e-10;
  constexpr double two_pi = 2.0 * 3.141592653589793;

  const auto fwd = specfem::fk_plugin::fft::make_fft_tables(kPower, 1.0);

  std::vector<double> r(kLx);
  for (int n = 0; n < kLx; ++n) {
    r[n] = std::sin(two_pi * k0 * static_cast<double>(n) / kLx);
  }

  auto buf = fft_tests_impl::to_complex(r);
  specfem::fk_plugin::fft::transform(kPower, buf.data(), 1.0, 1.0,
                                     fwd.bit_reversal.data(),
                                     fwd.twiddles.data());

  const double half = 0.5 * kLx;
  for (int k = 0; k < kLx; ++k) {
    const double expected_imag = (k == k0)         ? -half
                                 : (k == kLx - k0) ? half
                                                   : 0.0;
    EXPECT_NEAR(buf[k].real(), 0.0, kTol) << "real part at k=" << k;
    EXPECT_NEAR(buf[k].imag(), expected_imag, kTol) << "imag part at k=" << k;
  }
}

// ---------------------------------------------------------------------------
// make_fft_tables rejects powers outside [1, 30]
// ---------------------------------------------------------------------------

TEST(Fft, MakeTablesRejectsBadPower) {
  EXPECT_THROW(specfem::fk_plugin::fft::make_fft_tables(0, -1.0),
               std::runtime_error);
  EXPECT_THROW(specfem::fk_plugin::fft::make_fft_tables(-1, -1.0),
               std::runtime_error);
  EXPECT_THROW(specfem::fk_plugin::fft::make_fft_tables(31, -1.0),
               std::runtime_error);
}
