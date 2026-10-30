#include "specfem/fk_plugin/utilities/fft.hpp"
#include <gtest/gtest.h>
#include <pocketfft_hdronly.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Convention (unchanged from the in-house radix-2 FFT this replaced):
// forward kernel e^{-2pi*i*n*k/lx} scaled by dt, inverse kernel
// e^{+2pi*i*n*k/lx} scaled by 1/(lx*dt). inverse_real zeroes the Nyquist bin
// and makes DC real before transforming.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Forward (pocketfft r2c, scale dt) → inverse_real on an asymmetric signal
// recovers r[n] exactly
// ---------------------------------------------------------------------------

TEST(Fft, RoundTripAsymmetric) {
  constexpr int kPower = 6; // lx = 64
  constexpr int kLx = 1 << kPower;
  constexpr double kDt = 0.01;
  constexpr double kTol = 1.0e-10;
  constexpr double two_pi = 2.0 * std::numbers::pi;

  // Real, band-limited (no Nyquist energy), not time-reversal invariant.
  std::vector<double> r(kLx);
  for (int n = 0; n < kLx; ++n) {
    const double x = two_pi * static_cast<double>(n) / kLx;
    r[n] = 0.3 + std::cos(x) + 0.5 * std::sin(3.0 * x + 0.7) +
           0.2 * std::sin(5.0 * x);
  }

  std::vector<std::complex<double>> spectrum(kLx / 2 + 1);
  pocketfft::r2c<double>({ kLx }, { sizeof(double) },
                         { sizeof(std::complex<double>) }, 0,
                         pocketfft::FORWARD, r.data(), spectrum.data(), kDt);

  std::vector<double> out(kLx);
  specfem::fk_plugin::fft::inverse_real(spectrum.data(), kPower, kDt,
                                        out.data());

  for (int n = 0; n < kLx; ++n) {
    EXPECT_NEAR(out[n], r[n], kTol) << "mismatch at n=" << n;
  }
}

// ---------------------------------------------------------------------------
// Inverse kernel sign: S[k0] = -i lx dt/2 → x[n] = +sin(2π k0 n / lx)
// ---------------------------------------------------------------------------

TEST(Fft, InverseBinSign) {
  constexpr int kPower = 5; // lx = 32
  constexpr int kLx = 1 << kPower;
  constexpr int k0 = 3;
  constexpr double kDt = 0.5;
  constexpr double kTol = 1.0e-12;
  constexpr double two_pi = 2.0 * std::numbers::pi;

  std::vector<std::complex<double>> spectrum(kLx / 2 + 1);
  spectrum[k0] = std::complex<double>(0.0, -0.5 * kLx * kDt);

  std::vector<double> out(kLx);
  specfem::fk_plugin::fft::inverse_real(spectrum.data(), kPower, kDt,
                                        out.data());

  for (int n = 0; n < kLx; ++n) {
    EXPECT_NEAR(out[n], std::sin(two_pi * k0 * static_cast<double>(n) / kLx),
                kTol)
        << "mismatch at n=" << n;
  }
}

// ---------------------------------------------------------------------------
// inverse_real reproduces the reviewed radix-2 inverse
//
// Expected values were produced by the in-house FFT (make_fft_tables(power,
// -1) + inverse) before the switch to pocketfft. The spectrum has a complex
// DC, a non-zero Nyquist and garbage above Nyquist, so the values also pin
// the input contract: only bins [0, lx/2] are read, Nyquist zeroed, DC real.
// ---------------------------------------------------------------------------

namespace fft_tests_impl {

struct ReferenceSample {
  int power;
  double dt;
  int index;
  double value;
};

constexpr ReferenceSample reference_samples[] = {
  { 3, 1.0, 0, 0.3090736722891394 },
  { 3, 1.0, 1, 0.18425190102705896 },
  { 3, 1.0, 2, -0.022964542775281427 },
  { 3, 1.0, 3, 0.040488813764637588 },
  { 3, 1.0, 4, 0.053242669780009882 },
  { 3, 1.0, 5, -0.35391012274329875 },
  { 3, 1.0, 6, 0.40611527417365478 },
  { 3, 1.0, 7, 0.87463648141912476 },
  { 3, 0.01, 0, 30.907367228913941 },
  { 3, 0.01, 1, 18.425190102705898 },
  { 3, 0.01, 2, -2.2964542775281425 },
  { 3, 0.01, 3, 4.0488813764637586 },
  { 3, 0.01, 4, 5.3242669780009884 },
  { 3, 0.01, 5, -35.391012274329874 },
  { 3, 0.01, 6, 40.61152741736548 },
  { 3, 0.01, 7, 87.463648141912472 },
  { 6, 1.0, 0, 0.31753410136296983 },
  { 6, 1.0, 1, 0.024727903192144941 },
  { 6, 1.0, 2, 0.036815512085218972 },
  { 6, 1.0, 5, -0.1189399748826564 },
  { 6, 1.0, 16, 0.078652740934842258 },
  { 6, 1.0, 32, 0.0092207165273804348 },
  { 6, 1.0, 48, -0.15017609973752988 },
  { 6, 1.0, 63, 0.030301797198131253 },
  { 6, 0.01, 0, 31.753410136296985 },
  { 6, 0.01, 1, 2.4727903192144942 },
  { 6, 0.01, 2, 3.6815512085218973 },
  { 6, 0.01, 5, -11.89399748826564 },
  { 6, 0.01, 16, 7.865274093484226 },
  { 6, 0.01, 32, 0.92207165273804348 },
  { 6, 0.01, 48, -15.017609973752988 },
  { 6, 0.01, 63, 3.0301797198131251 },
  { 10, 1.0, 0, 0.016250564032286925 },
  { 10, 1.0, 1, 0.039478450385623295 },
  { 10, 1.0, 2, -0.014427266670820534 },
  { 10, 1.0, 5, 0.05786759126855575 },
  { 10, 1.0, 256, -0.038568591016030447 },
  { 10, 1.0, 512, -0.0059343141708844103 },
  { 10, 1.0, 768, 0.013522292725387653 },
  { 10, 1.0, 1023, 0.018487033601823369 },
  { 10, 0.01, 0, 1.6250564032286925 },
  { 10, 0.01, 1, 3.9478450385623294 },
  { 10, 0.01, 2, -1.4427266670820533 },
  { 10, 0.01, 5, 5.7867591268555749 },
  { 10, 0.01, 256, -3.8568591016030447 },
  { 10, 0.01, 512, -0.59343141708844105 },
  { 10, 0.01, 768, 1.3522292725387652 },
  { 10, 0.01, 1023, 1.8487033601823368 },
};

// Spectrum the reference values were recorded with.
std::vector<std::complex<double>> reference_spectrum(int power) {
  const int lx = 1 << power;
  const int nhalf = lx / 2;
  std::vector<std::complex<double>> spectrum(lx);
  for (int k = 0; k <= nhalf; ++k) {
    const double amp = 1.0 + 0.5 * std::cos(0.7 * k);
    const double phi = 0.37 * k * k + 0.11;
    spectrum[k] =
        std::complex<double>(amp * std::cos(phi), amp * std::sin(phi));
  }
  for (int k = nhalf + 1; k < lx; ++k) {
    spectrum[k] = std::complex<double>(99.0, -99.0);
  }
  return spectrum;
}

} // namespace fft_tests_impl

TEST(Fft, InverseMatchesReviewedValues) {
  constexpr double kRelTol = 1.0e-12;

  for (const auto &sample : fft_tests_impl::reference_samples) {
    auto spectrum = fft_tests_impl::reference_spectrum(sample.power);
    const int lx = 1 << sample.power;

    // Roundoff scales with the summed terms, not with the (smaller) result:
    // |x_n| <= 2/(lx dt) * sum_{k<=lx/2} |S_k|.
    double bound = 0.0;
    for (int k = 0; k <= lx / 2; ++k) {
      bound += std::abs(spectrum[k]);
    }
    bound *= 2.0 / (lx * sample.dt);

    std::vector<double> out(lx);
    specfem::fk_plugin::fft::inverse_real(spectrum.data(), sample.power,
                                          sample.dt, out.data());

    EXPECT_NEAR(out[sample.index], sample.value, kRelTol * bound)
        << "power=" << sample.power << " dt=" << sample.dt
        << " n=" << sample.index;
  }
}

// ---------------------------------------------------------------------------
// inverse_real rejects powers outside [1, 30]
// ---------------------------------------------------------------------------

TEST(Fft, InverseRealRejectsBadPower) {
  std::vector<std::complex<double>> spectrum(2);
  std::vector<double> out(2);
  EXPECT_THROW(specfem::fk_plugin::fft::inverse_real(spectrum.data(), 0, 1.0,
                                                     out.data()),
               std::runtime_error);
  EXPECT_THROW(specfem::fk_plugin::fft::inverse_real(spectrum.data(), -1, 1.0,
                                                     out.data()),
               std::runtime_error);
  EXPECT_THROW(specfem::fk_plugin::fft::inverse_real(spectrum.data(), 31, 1.0,
                                                     out.data()),
               std::runtime_error);
}
