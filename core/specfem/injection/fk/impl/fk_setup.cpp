#include "specfem/injection/fk/impl/fk_setup.hpp"

#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/injection/fk/impl/layer_operators.hpp"

#include <cmath>

specfem::injection::fk_impl::FkSetup specfem::injection::fk_impl::prepare(
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk::IncidentWave &wave,
    const specfem::injection::fk::TimeWindow &window, double p,
    const specfem::injection::fk_impl::DepthGroups &groups,
    specfem::injection::fk::field_derivative derivative,
    bool compute_traction) {
  specfem::injection::fk_impl::FkSetup setup;

  const double span = groups.time_delay_max - groups.time_delay_min;
  setup.sizes = specfem::injection::fk::compute_fk_sizes(window, span);
  setup.fft_power = setup.sizes.interpolation_power;
  setup.p = p;
  setup.phi = static_cast<double>(wave.azimuth_phi);
  setup.dt_min = static_cast<double>(setup.sizes.resampling_rate) *
                 static_cast<double>(window.dt);
  setup.compute_traction = compute_traction;
  setup.wave_type = wave.type;
  setup.derivative_order =
      (derivative == specfem::injection::fk::field_derivative::velocity) ? 1
      : (derivative == specfem::injection::fk::field_derivative::acceleration)
          ? 2
          : 0;

  // Positive-frequency samples (including DC): nf_stored + 1.
  const int nf2 = setup.sizes.number_of_stored_frequencies + 1;
  const double df = static_cast<double>(setup.sizes.frequency_step);
  const double half_duration = static_cast<double>(wave.gaussian_half_duration);

  setup.omega.resize(nf2);
  setup.apodization.resize(nf2);
  for (int i = 0; i < nf2; ++i) {
    setup.omega[i] =
        2.0 * specfem::injection::fk_impl::pi * static_cast<double>(i) * df;
    const double x = setup.omega[i] * half_duration / 2.0;
    setup.apodization[i] = std::exp(-x * x);
  }

  // t0 / delay bookkeeping (see
  // injection_plan/rework/14-time-placement-bookkeeping.md). nn =
  // int(origin_time / dt_min) (truncation toward zero); pad keeps per-point
  // sample abscissae non-negative; the stored series is left-rotated by nn -
  // pad.
  const int nn =
      static_cast<int>(static_cast<double>(wave.origin_time) / setup.dt_min);
  setup.pad_samples =
      static_cast<int>(std::ceil(groups.time_delay_max / setup.dt_min));
  setup.shift_samples = nn - setup.pad_samples;

  const specfem::injection::fk::ElasticIsotropicLayer &halfspace =
      model.elastic_layers().back();
  setup.e_mat = specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(
      halfspace, static_cast<type_real>(p));
  setup.amplitude = specfem::injection::fk_impl::incident_amplitude(
      halfspace, wave.type, wave.amplitude, static_cast<type_real>(p));

  return setup;
}
