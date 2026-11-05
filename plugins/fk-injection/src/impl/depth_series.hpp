#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/layered_model.hpp"
#include "impl/depth_groups.hpp"
#include "impl/fk_setup.hpp"
#include "impl/layer_stack.hpp"
#include "impl/series_layout.hpp"
#include <complex>
#include <cstddef>
#include <vector>

namespace fk_injection::impl {

/**
 * @brief Cubic-B-spline coefficients of the depth-only series, per group.
 *
 * Laid out as @c series_coefficient_index describes, so each series row is
 * contiguous for spline evaluation. The series slots follow the @c series_*
 * indices (S0..S4).
 */
struct DepthSeries {
  std::vector<fk_injection::type_real> coefficients{};
  int number_of_groups = 0;
  int coefficient_count = 0;
};

/**
 * @brief Fill the positive-frequency depth spectrum for one group.
 *
 * For every stored positive frequency: evaluate the state at the group depth
 * (half-space or partial-layer propagation via the layer operators), project to
 * the depth-only series S0..S4, multiply by the Gaussian apodization, and apply
 * the (i*omega)^n derivative factor to S0/S1 only. The free-surface groups get
 * their fluid pressure zeroed before projection.
 *
 * @param group      Location of the group's depth.
 * @param states     Interface states for all frequencies.
 * @param model      Layered velocity model.
 * @param setup      Shared frequency-domain setup.
 * @param spectrum   Output buffer, laid out as [number_of_series][fft_length];
 *                   bins [0, nf_stored] are written per series.
 * @param fft_length Stride between series (the FFT length N).
 */
void depth_spectrum(const fk_injection::LayerLocation &group,
                    const InterfaceStates &states,
                    const fk_injection::LayeredModel &model,
                    const FkSetup &setup, std::complex<double> *spectrum,
                    int fft_length);

/**
 * @brief Convert a group's depth spectrum to stored B-spline coefficients.
 *
 * Per series: inverse FFT (Hermitian completion handled internally),
 * left-rotate by @c setup.shift_samples for t0 alignment, cubic-B-spline
 * prefilter over all N samples, keep the first @c coefficient_count, and
 * cosine-taper the leading 20 coefficients. The coefficients are computed in
 * double and narrowed to @c type_real on write.
 *
 * The inverse FFT divides by @c setup.normalization_dt.
 *
 * @param spectrum     Group spectrum [number_of_series][N] (modified in place).
 * @param setup        Shared setup (sizes, shift, normalization).
 * @param coefficients Output [number_of_series][coefficient_count], contiguous.
 */
void to_time_series(std::complex<double> *spectrum, const FkSetup &setup,
                    fk_injection::type_real *coefficients);

/**
 * @brief Synthesize the depth-only series for every group.
 *
 * Host loop over groups; per group it fills the spectrum and converts it to
 * stored coefficients back-to-back (spectra are not all materialized at once).
 *
 * @param groups Depth groups.
 * @param states Interface states.
 * @param model  Layered velocity model.
 * @param setup  Shared frequency-domain setup.
 * @return DepthSeries with per-group coefficients and the t0/pad bookkeeping.
 */
DepthSeries depth_series(const DepthGroups &groups,
                         const InterfaceStates &states,
                         const fk_injection::LayeredModel &model,
                         const FkSetup &setup);

} // namespace fk_injection::impl
