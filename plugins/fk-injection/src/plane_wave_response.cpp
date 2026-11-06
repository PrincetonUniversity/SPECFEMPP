#include "plane_wave_response.hpp"

#include "impl/fk_math_impl.hpp"
#include "impl/halfspace_solve.hpp"
#include "impl/layer_operators.hpp"
#include "specfem/plugin/injection_plugin.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <variant>

double fk_injection::impl::fluid_reverberation_time(
    const fk_injection::LayeredModel &model, double p) {
  static constexpr double tolerance = 1.0e-4;
  const auto &layers = model.layers();
  int fluid_count = 0;
  while (
      std::holds_alternative<fk_injection::AcousticLayer>(layers[fluid_count]))
    ++fluid_count;
  if (fluid_count == 0)
    return 0.0;

  const auto vertical_slowness = [p](double velocity) {
    return std::sqrt(
        std::complex<double>(1.0 / (velocity * velocity) - p * p, 0.0));
  };

  double round_trip = 0.0;
  for (int i = 0; i < fluid_count; ++i) {
    const auto &layer = std::get<fk_injection::AcousticLayer>(layers[i]);
    const double velocity = static_cast<double>(layer.p_velocity);
    // Evanescent in the fluid: nothing propagates to reverberate.
    if (p * velocity >= 1.0)
      return 0.0;
    round_trip += 2.0 * static_cast<double>(layer.thickness) *
                  vertical_slowness(velocity).real();
  }

  // Fluid-to-solid P reflection with impedances rho / q (Brekhovskikh); the
  // free surface reflects with -1, so |R| is the loss per round trip.
  const auto &fluid =
      std::get<fk_injection::AcousticLayer>(layers[fluid_count - 1]);
  const auto &solid =
      std::get<fk_injection::ElasticIsotropicLayer>(layers[fluid_count]);
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

void fk_injection::impl::check_supported_slowness(
    const fk_injection::LayeredModel &model, double p) {
  // The negated comparisons also reject NaN.
  if (!(p > 0.0)) {
    std::ostringstream message;
    message << "FK needs a positive ray parameter, got p = " << p
            << " s/m; vertical incidence is not supported";
    throw specfem::plugin::Unsupported(message.str());
  }

  double fastest = 0.0;
  for (const auto &layer : model.layers()) {
    if (const auto *solid =
            std::get_if<fk_injection::ElasticIsotropicLayer>(&layer)) {
      fastest = std::max({ fastest, static_cast<double>(solid->p_velocity),
                           static_cast<double>(solid->s_velocity) });
    } else {
      fastest = std::max(
          fastest,
          static_cast<double>(
              std::get<fk_injection::AcousticLayer>(layer).p_velocity));
    }
  }
  const double limit = 1.0 / fastest;
  if (!(p < limit)) {
    std::ostringstream message;
    message << "FK needs p below " << limit << " s/m (1/v for the fastest "
            << "velocity in the model, " << fastest << " m/s), got p = " << p
            << " s/m; critical and post-critical incidence are not supported, "
            << "which limits SV to take-off angles below asin(vs/vp)";
    throw specfem::plugin::Unsupported(message.str());
  }
}

fk_injection::PlaneWaveResponse::PlaneWaveResponse(
    const fk_injection::LayeredModel &model,
    const fk_injection::IncidentWave &wave,
    const fk_injection::TimeWindow &window,
    const fk_injection::HorizontalBox &box,
    const fk_injection::ReferenceConvention convention)
    : model_(model), ray_parameter_(wave.ray_parameter(model)),
      azimuth_(static_cast<double>(wave.azimuth_phi)),
      origin_x_(static_cast<double>(wave.origin_x)),
      origin_y_(static_cast<double>(wave.origin_y)),
      number_of_steps_(window.nstep) {
  const double p = ray_parameter_;
  fk_injection::impl::check_supported_slowness(model, p);

  const double vp_halfspace = static_cast<double>(model.halfspace().p_velocity);
  const double eta_p = std::sqrt(1.0 / (vp_halfspace * vp_halfspace) - p * p);
  vertical_delay_ = eta_p * (-model.halfspace_top_depth() -
                             static_cast<double>(wave.origin_z));

  // The delay is linear in x and y, so the corners bound it over the box.
  const double corners[4][2] = { { box.x_min, box.y_min },
                                 { box.x_min, box.y_max },
                                 { box.x_max, box.y_min },
                                 { box.x_max, box.y_max } };
  double time_delay_min = std::numeric_limits<double>::infinity();
  double time_delay_max = -std::numeric_limits<double>::infinity();
  for (const auto &[x, y] : corners) {
    time_delay_min = std::min(time_delay_min, time_delay(x, y));
    time_delay_max = std::max(time_delay_max, time_delay(x, y));
  }

  setup_ = prepare(model, wave, window, p, time_delay_min, time_delay_max,
                   convention);

  for (const auto &[x, y] : corners)
    check_coverage(x, y);

  // The series are periodic in the FFT length, so FK times outside one period
  // would silently read another part of the field.
  const double resampling = static_cast<double>(setup_.sizes.resampling_rate);
  const double half_period =
      0.5 * static_cast<double>(setup_.sizes.number_of_fft_samples);
  const double origin_samples =
      static_cast<double>(wave.origin_time) / setup_.dt_min;
  const double first_sample = origin_samples - time_delay_max / setup_.dt_min;
  const double last_sample =
      origin_samples + static_cast<double>(window.nstep - 1) / resampling -
      time_delay_min / setup_.dt_min;
  if (first_sample < -half_period || last_sample >= half_period) {
    std::ostringstream message;
    message << "fk_injection: the run needs FK times from "
            << first_sample * setup_.dt_min << " to "
            << last_sample * setup_.dt_min << " s, outside the FFT period of "
            << 2.0 * half_period * setup_.dt_min
            << " s centered on 0; move origin-time closer to the run or "
               "increase the injection time-window length";
    throw std::runtime_error(message.str());
  }

  states_ = fk_injection::impl::solve_layer_stack(model, setup_);
}

double fk_injection::PlaneWaveResponse::time_delay(const double x,
                                                   const double y) const {
  return ray_parameter_ * ((x - origin_x_) * std::cos(azimuth_) +
                           (y - origin_y_) * std::sin(azimuth_)) +
         vertical_delay_;
}

double fk_injection::PlaneWaveResponse::abscissa_offset(const double x,
                                                        const double y) const {
  return setup_.abscissa_pad -
         time_delay(x, y) * setup_.delay_stretch / setup_.dt_min;
}

void fk_injection::PlaneWaveResponse::check_coverage(const double x,
                                                     const double y) const {
  const int ncoef = setup_.sizes.number_of_stored_frequencies;
  const double offset = setup_.abscissa_pad - time_delay(x, y) / setup_.dt_min;
  const double max_abscissa =
      static_cast<double>(number_of_steps_ - 1) /
          static_cast<double>(setup_.sizes.resampling_rate) +
      offset;
  if (offset < 0.0 || max_abscissa > static_cast<double>(ncoef - 1)) {
    std::ostringstream message;
    message << "fk_injection: sampling abscissa out of range at (" << x << ", "
            << y << ") (offset " << offset << ", max " << max_abscissa
            << ", coefficients " << ncoef
            << "); increase the injection time-window length";
    throw std::runtime_error(message.str());
  }
}

fk_injection::impl::FkSetup fk_injection::PlaneWaveResponse::prepare(
    const fk_injection::LayeredModel &model,
    const fk_injection::IncidentWave &wave,
    const fk_injection::TimeWindow &window, double p, double time_delay_min,
    double time_delay_max, fk_injection::ReferenceConvention convention) {
  fk_injection::impl::FkSetup setup;
  const bool specfem3d =
      convention == fk_injection::ReferenceConvention::specfem3d;

  const double span = time_delay_max - time_delay_min;
  // The series repeat with the FFT period, so a reverberation that outlasts it
  // wraps onto the start of the run. SPECFEM3D's window has no such floor.
  const double reverberation =
      specfem3d ? 0.0 : fk_injection::impl::fluid_reverberation_time(model, p);
  const double minimum_window =
      reverberation > 0.0 ? span + reverberation : 0.0;
  setup.sizes = fk_injection::compute_fk_sizes(window, span, minimum_window);
  setup.p = p;
  setup.phi = static_cast<double>(wave.azimuth_phi);
  setup.dt_min = static_cast<double>(setup.sizes.resampling_rate) *
                 static_cast<double>(window.dt);
  setup.wave_type = wave.type;

  const double fft_samples =
      static_cast<double>(setup.sizes.number_of_fft_samples);
  setup.delay_stretch = specfem3d ? fft_samples / (fft_samples - 1.0) : 1.0;
  setup.normalization_dt =
      specfem3d ? static_cast<double>(window.dt) : setup.dt_min;

  // Positive-frequency samples (including DC): nf_stored + 1.
  const int nf2 = setup.sizes.number_of_stored_frequencies + 1;
  // The inverse FFT assumes 1 / (N dt_min); SPECFEM3D's grid uses N - 1.
  const double df = specfem3d ? static_cast<double>(setup.sizes.frequency_step)
                              : 1.0 / (fft_samples * setup.dt_min);
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
  const int nn = static_cast<int>(specfem3d ? std::trunc(origin_samples)
                                            : std::floor(origin_samples));
  setup.pad_samples =
      static_cast<int>(std::ceil(time_delay_max / setup.dt_min));
  setup.shift_samples = nn - setup.pad_samples;
  setup.abscissa_pad = static_cast<double>(setup.pad_samples);
  if (!specfem3d)
    setup.abscissa_pad += origin_samples - static_cast<double>(nn);

  const fk_injection::ElasticIsotropicLayer &halfspace = model.halfspace();
  setup.e_mat = fk_injection::impl::elastic_halfspace_eigenmatrix(halfspace, p);
  setup.amplitude = fk_injection::impl::incident_amplitude(halfspace, wave.type,
                                                           wave.amplitude, p);

  return setup;
}
