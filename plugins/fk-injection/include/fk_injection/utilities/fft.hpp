#pragma once

#include <pocketfft_hdronly.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace fk_injection {
namespace fft {

/**
 * @brief Normalization of @c irfft, as in @c numpy.fft.
 */
enum class norm {
  backward, ///< Scale by \f$ 1/n \f$.
  ortho,    ///< Scale by \f$ 1/\sqrt{n} \f$.
  forward   ///< No scaling.
};

} // namespace fft

namespace fft_impl {

/**
 * @brief Throw unless @p half_spectrum_size bins cover an output of
 * @p output_size samples.
 * @param half_spectrum_size Number of complex input bins.
 * @param output_size Transform length n.
 * @throws std::invalid_argument if n is zero or fewer than n/2 + 1 bins.
 */
inline void check_sizes(const std::size_t half_spectrum_size,
                        const std::size_t output_size) {
  if (output_size == 0) {
    throw std::invalid_argument(
        "fk_injection::fft: output length must be positive");
  }
  if (half_spectrum_size < output_size / 2 + 1) {
    throw std::invalid_argument("fk_injection::fft: spectrum needs at "
                                "least n/2 + 1 bins for n output samples");
  }
}

/**
 * @brief Hermitian complex-to-real transform of length @c out.size(), scaled
 * by @p factor.
 * @param spectrum Bins @c [0, n/2] (at least); higher bins are not read.
 * @param out Output samples; its size is the transform length n.
 * @param factor Scale applied to every output sample.
 */
inline void complex_to_real(std::span<const std::complex<double>> spectrum,
                            std::span<double> out, const double factor) {
  check_sizes(spectrum.size(), out.size());
  pocketfft::c2r<double>({ out.size() }, { sizeof(std::complex<double>) },
                         { sizeof(double) }, 0, pocketfft::BACKWARD,
                         spectrum.data(), out.data(), factor);
}

} // namespace fft_impl

namespace fft {

/**
 * @brief Inverse of the real FFT, shaped like @c numpy.fft.irfft.
 *
 * With @c norm::backward computes
 * \f[
 *   x_m = \frac{1}{n} \sum_{k=0}^{n-1} a_k\, e^{+2\pi i m k / n},
 *   \qquad a_{n-k} = \overline{a_k},
 * \f]
 * for any length n (not only powers of two). Only bins @c [0, n/2] are read.
 * As in NumPy, the imaginary parts of the DC bin and (for even n) the Nyquist
 * bin are ignored; the Nyquist real part is used.
 *
 * @param a Half spectrum, at least @c n/2 + 1 bins; not modified.
 * @param out Output samples; its size is the transform length n.
 * @param normalization Output scaling (see @c norm).
 * @throws std::invalid_argument if @c out is empty or @p a has fewer than
 *         @c n/2 + 1 bins.
 */
inline void irfft(std::span<const std::complex<double>> a,
                  std::span<double> out,
                  const norm normalization = norm::backward) {
  const double n = static_cast<double>(out.size());
  const double factor = normalization == norm::backward ? 1.0 / n
                        : normalization == norm::ortho  ? 1.0 / std::sqrt(n)
                                                        : 1.0;
  fft_impl::complex_to_real(a, out, factor);
}

/**
 * @brief FK inverse transform: @c irfft with the Nyquist bin dropped and the
 * forward transform's @p dt undone.
 *
 * Computes
 * \f[
 *   x_m = \frac{1}{n\,\Delta t} \sum_{k=0}^{n-1} S_k\, e^{+2\pi i m k / n},
 * \f]
 * with the Nyquist bin (even n) taken as zero and the DC bin as real. Works on
 * a copy, so @p spectrum is not modified.
 *
 * @param spectrum Half spectrum, at least @c n/2 + 1 bins.
 * @param dt Time step of the forward transform this inverts.
 * @param out Output samples; its size is the transform length n.
 * @throws std::invalid_argument if @c out is empty or @p spectrum has fewer
 *         than @c n/2 + 1 bins.
 */
inline void inverse_real(std::span<const std::complex<double>> spectrum,
                         const double dt, std::span<double> out) {
  const std::size_t n = out.size();
  fft_impl::check_sizes(spectrum.size(), n);

  std::vector<std::complex<double>> half_spectrum(
      spectrum.begin(), spectrum.begin() + (n / 2 + 1));
  if (n % 2 == 0) {
    half_spectrum[n / 2] = std::complex<double>(0.0, 0.0);
  }

  fft_impl::complex_to_real(half_spectrum, out,
                            1.0 / (static_cast<double>(n) * dt));
}

} // namespace fft
} // namespace fk_injection
