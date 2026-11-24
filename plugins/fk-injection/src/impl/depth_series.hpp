#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/layered_model.hpp"
#include "impl/layer_stack.hpp"
#include <complex>
#include <cstddef>
#include <vector>

namespace fk_injection::impl {

struct FkSetup; // defined with PlaneWaveResponse

/**
 * @brief Fill the positive-frequency depth spectrum at one location.
 *
 * For every stored positive frequency: evaluate the state at the depth
 * (half-space or partial-layer propagation via the layer operators), project to
 * the series of the location's medium, multiply by the Gaussian apodization,
 * and apply i*omega to the elastic horizontal and vertical series, which makes
 * them velocity. Free-surface
 * locations get their fluid pressure zeroed before projection.
 *
 * @param location   Location of the depth.
 * @param states     Interface states for all frequencies.
 * @param model      Layered velocity model.
 * @param setup      Shared frequency-domain setup.
 * @param spectrum   Output buffer, laid out as [series][fft_length] with
 *                   @c number_of_series(location.medium) series; bins
 *                   [0, nf_stored] are written per series.
 * @param fft_length Stride between series (the FFT length N).
 */
void depth_spectrum(const fk_injection::LayerLocation &location,
                    const InterfaceStates &states,
                    const fk_injection::LayeredModel &model,
                    const FkSetup &setup, std::complex<double> *spectrum,
                    int fft_length);

/**
 * @brief Convert a depth spectrum to stored B-spline coefficients.
 *
 * Per series: inverse FFT (Hermitian completion handled internally),
 * left-rotate by @c setup.shift_samples for t0 alignment, cubic-B-spline
 * prefilter over all N samples, keep the first @c coefficient_count, and
 * cosine-taper the leading 20 coefficients. The coefficients are computed in
 * double and narrowed to @c type_real on write.
 *
 * The inverse FFT divides by @c setup.normalization_dt.
 *
 * @param spectrum         Spectrum [number_of_series][N] (modified in place).
 * @param setup            Shared setup (sizes, shift, normalization).
 * @param number_of_series Series in @p spectrum.
 * @param coefficients     Output [number_of_series][coefficient_count],
 *                         contiguous.
 */
void to_time_series(std::complex<double> *spectrum, const FkSetup &setup,
                    int number_of_series,
                    fk_injection::type_real *coefficients);

} // namespace fk_injection::impl
