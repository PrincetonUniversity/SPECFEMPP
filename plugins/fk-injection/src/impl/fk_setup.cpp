#include "impl/fk_setup.hpp"

#include "impl/fk_math_impl.hpp"
#include "impl/layer_operators.hpp"

#include <cmath>

fk_injection::impl::FkSetup
fk_injection::impl::prepare(const fk_injection::LayeredModel &model,
                            const fk_injection::IncidentWave &wave,
                            const fk_injection::TimeWindow &window, double p,
                            const fk_injection::impl::DepthGroups &groups,
                            fk_injection::field_derivative derivative,
                            fk_injection::ReferenceConvention convention) {
  fk_injection::impl::FkSetup setup;

  const double span = groups.time_delay_max - groups.time_delay_min;
  setup.sizes = fk_injection::compute_fk_sizes(window, span);
  setup.fft_power = setup.sizes.interpolation_power;
  setup.p = p;
  setup.phi = static_cast<double>(wave.azimuth_phi);
  setup.dt_min = static_cast<double>(setup.sizes.resampling_rate) *
                 static_cast<double>(window.dt);
  setup.wave_type = wave.type;
  setup.derivative = derivative;

  // Positive-frequency samples (including DC): nf_stored + 1.
  const int nf2 = setup.sizes.number_of_stored_frequencies + 1;
  // The inverse FFT assumes 1 / (N dt_min); SPECFEM3D's grid uses N - 1.
  const double df =
      convention == fk_injection::ReferenceConvention::specfem3d
          ? static_cast<double>(setup.sizes.frequency_step)
          : 1.0 / (static_cast<double>(setup.sizes.number_of_fft_samples) *
                   setup.dt_min);
  const double half_duration = static_cast<double>(wave.gaussian_half_duration);

  setup.omega.resize(nf2);
  setup.apodization.resize(nf2);
  for (int i = 0; i < nf2; ++i) {
    setup.omega[i] = 2.0 * fk_injection::impl::pi * static_cast<double>(i) * df;
    const double x = setup.omega[i] * half_duration / 2.0;
    setup.apodization[i] = std::exp(-x * x);
  }

  // t0 / delay bookkeeping. pad keeps per-point sample abscissae non-negative;
  // the stored series is left-rotated by nn - pad. Flooring leaves a fraction
  // in [0, 1) for the abscissa, so the abscissae stay non-negative too.
  const double origin_samples =
      static_cast<double>(wave.origin_time) / setup.dt_min;
  const bool truncate =
      convention == fk_injection::ReferenceConvention::specfem3d;
  const int nn = static_cast<int>(truncate ? std::trunc(origin_samples)
                                           : std::floor(origin_samples));
  setup.pad_samples =
      static_cast<int>(std::ceil(groups.time_delay_max / setup.dt_min));
  setup.shift_samples = nn - setup.pad_samples;
  setup.abscissa_pad = static_cast<double>(setup.pad_samples);
  if (!truncate)
    setup.abscissa_pad += origin_samples - static_cast<double>(nn);

  const fk_injection::ElasticIsotropicLayer &halfspace =
      model.elastic_layers().back();
  setup.e_mat = fk_injection::impl::elastic_halfspace_eigenmatrix(halfspace, p);
  setup.amplitude = fk_injection::impl::incident_amplitude(halfspace, wave.type,
                                                           wave.amplitude, p);

  return setup;
}
