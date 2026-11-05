#include "plane_wave_response.hpp"
#include "impl/driver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

fk_injection::HorizontalBox
fk_injection::bounding_box(const std::vector<fk_injection::type_real> &x,
                           const std::vector<fk_injection::type_real> &y) {
  if (x.empty())
    return {};
  const auto [x_min, x_max] = std::minmax_element(x.begin(), x.end());
  const auto [y_min, y_max] = std::minmax_element(y.begin(), y.end());
  return { static_cast<double>(*x_min), static_cast<double>(*x_max),
           static_cast<double>(*y_min), static_cast<double>(*y_max) };
}

fk_injection::PlaneWaveResponse::PlaneWaveResponse(
    const fk_injection::LayeredModel &model,
    const fk_injection::IncidentWave &wave,
    const fk_injection::TimeWindow &window,
    const fk_injection::HorizontalBox &box,
    const fk_injection::ReferenceConvention convention)
    : model_(model), origin_x_(static_cast<double>(wave.origin_x)),
      origin_y_(static_cast<double>(wave.origin_y)),
      number_of_steps_(window.nstep) {
  const double p = wave.ray_parameter(model);
  fk_injection::impl::check_supported_slowness(model, p);

  const double vp_halfspace =
      static_cast<double>(model.elastic_layers().back().p_velocity);
  const double eta_p = std::sqrt(1.0 / (vp_halfspace * vp_halfspace) - p * p);
  vertical_delay_ = eta_p * (-model.halfspace_top_depth() -
                             static_cast<double>(wave.origin_z));
  // time_delay needs the ray parameter and azimuth before prepare runs.
  setup_.p = p;
  setup_.phi = static_cast<double>(wave.azimuth_phi);

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

  setup_ = fk_injection::impl::prepare(model, wave, window, p, time_delay_min,
                                       time_delay_max, convention);

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
  return setup_.p * ((x - origin_x_) * std::cos(setup_.phi) +
                     (y - origin_y_) * std::sin(setup_.phi)) +
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
