#pragma once

#include <pocketfft_hdronly.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>

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
  const std::size_t n = out.size();
  if (n == 0) {
    throw std::invalid_argument(
        "fk_injection::fft::irfft: output length must be positive");
  }
  if (a.size() < n / 2 + 1) {
    throw std::invalid_argument("fk_injection::fft::irfft: spectrum "
                                "needs at least n/2 + 1 bins for n samples");
  }

  const double length = static_cast<double>(n);
  const double factor = normalization == norm::backward ? 1.0 / length
                        : normalization == norm::ortho ? 1.0 / std::sqrt(length)
                                                       : 1.0;

  pocketfft::c2r<double>({ n }, { sizeof(std::complex<double>) },
                         { sizeof(double) }, 0, pocketfft::BACKWARD, a.data(),
                         out.data(), factor);
}

} // namespace fft
} // namespace fk_injection
