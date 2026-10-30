#pragma once

#include <pocketfft_hdronly.h>

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace specfem {
namespace fk_plugin {
namespace fft {

/**
 * @brief Real inverse FFT of a Hermitian half-spectrum (pocketfft).
 *
 * Computes
 * @f[
 *   x_n = \frac{1}{l_x\,\Delta t} \sum_{k=0}^{l_x-1} S_k\,
 *         e^{+2\pi i n k / l_x}, \qquad l_x = 2^{power},
 * @f]
 * with @f$ S_{l_x-k} = \overline{S_k} @f$. Only bins @c [0, lx/2] are read.
 * The Nyquist bin @c spectrum[lx/2] is zeroed and the DC bin made real
 * (both in place) before the transform.
 *
 * @param spectrum Complex spectrum of length @c lx/2 + 1 or more; bins 0 and
 *                 @c lx/2 are modified.
 * @param power    log2 of the transform length; must be in [1, 30].
 * @param dt       Time step of the forward transform this inverts.
 * @param out      Output buffer of length @c lx.
 * @throws std::runtime_error if @p power < 1 or @p power > 30.
 */
inline void inverse_real(std::complex<double> *spectrum, int power, double dt,
                         double *out) {
  if (power < 1 || power > 30) {
    throw std::runtime_error("specfem::fk_plugin::fft::inverse_real: power "
                             "must be in [1, 30]");
  }

  const std::size_t lx = std::size_t{ 1 } << power;
  spectrum[lx / 2] = std::complex<double>(0.0, 0.0);
  spectrum[0] = std::complex<double>(spectrum[0].real(), 0.0);

  pocketfft::c2r<double>({ lx }, { sizeof(std::complex<double>) },
                         { sizeof(double) }, 0, pocketfft::BACKWARD, spectrum,
                         out, 1.0 / (static_cast<double>(lx) * dt));
}

} // namespace fft
} // namespace fk_plugin
} // namespace specfem
