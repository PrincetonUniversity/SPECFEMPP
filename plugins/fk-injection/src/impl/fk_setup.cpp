#include "impl/fk_setup.hpp"

#include "impl/fk_math_impl.hpp"
#include "impl/layer_operators.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>

double fk_injection::impl::fluid_reverberation_time(
    const fk_injection::LayeredModel &model, double p) {
  static constexpr double tolerance = 1.0e-4;
  const auto &fluid_layers = model.fluid_layers();
  if (fluid_layers.empty())
    return 0.0;

  const auto vertical_slowness = [p](double velocity) {
    return std::sqrt(
        std::complex<double>(1.0 / (velocity * velocity) - p * p, 0.0));
  };

  double round_trip = 0.0;
  for (const auto &layer : fluid_layers) {
    const double velocity = static_cast<double>(layer.p_velocity);
    // Evanescent in the fluid: nothing propagates to reverberate.
    if (p * velocity >= 1.0)
      return 0.0;
    round_trip += 2.0 * static_cast<double>(layer.thickness) *
                  vertical_slowness(velocity).real();
  }

  // Fluid-to-solid P reflection with impedances rho / q (Brekhovskikh); the
  // free surface reflects with -1, so |R| is the loss per round trip.
  const fk_injection::AcousticLayer &fluid = fluid_layers.back();
  const fk_injection::ElasticIsotropicLayer &solid =
      model.elastic_layers().front();
  const double density = static_cast<double>(solid.density);
  const double s_velocity = static_cast<double>(solid.s_velocity);
  const double cos_2 = 1.0 - 2.0 * s_velocity * s_velocity * p * p;
  const std::complex<double> sin_2 =
      2.0 * s_velocity * s_velocity * p * vertical_slowness(s_velocity);
  const std::complex<double> solid_impedance =
      density / vertical_slowness(static_cast<double>(solid.p_velocity)) *
          cos_2 * cos_2 +
      density / vertical_slowness(s_velocity) * sin_2 * sin_2;
  const std::complex<double> fluid_impedance =
      static_cast<double>(fluid.density) /
      vertical_slowness(static_cast<double>(fluid.p_velocity));
  const double reflection = std::abs((solid_impedance - fluid_impedance) /
                                     (solid_impedance + fluid_impedance));
  if (!(reflection < 1.0))
    throw std::runtime_error(
        "fk_injection: the fluid layers reflect the incident wave totally, so "
        "their reverberation never decays");

  return round_trip * std::log(tolerance) / std::log(reflection);
}

fk_injection::impl::FkSetup
fk_injection::impl::prepare(const fk_injection::LayeredModel &model,
                            const fk_injection::IncidentWave &wave,
                            const fk_injection::TimeWindow &window, double p,
                            const fk_injection::impl::DepthGroups &groups,
                            fk_injection::field_derivative derivative,
                            fk_injection::ReferenceConvention convention) {
  fk_injection::impl::FkSetup setup;

  const double span = groups.time_delay_max - groups.time_delay_min;
  // The series repeat with the FFT period, so a reverberation that outlasts it
  // wraps onto the start of the run. SPECFEM3D's window has no such floor.
  const double reverberation =
      convention == fk_injection::ReferenceConvention::exact
          ? fk_injection::impl::fluid_reverberation_time(model, p)
          : 0.0;
  const double minimum_window =
      reverberation > 0.0 ? span + reverberation : 0.0;
  setup.sizes = fk_injection::compute_fk_sizes(window, span, minimum_window);
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
