#include "fk_injection/solver.hpp"
#include "impl/depth_groups.hpp"
#include "impl/depth_series.hpp"
#include "impl/driver.hpp"
#include "impl/fk_setup.hpp"
#include "impl/layer_stack.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

fk_injection::FkResult
fk_injection::solve(const fk_injection::LayeredModel &model,
                    const fk_injection::IncidentWave &wave,
                    const fk_injection::TimeWindow &window,
                    const fk_injection::EvalPoints &points,
                    fk_injection::field_derivative derivative,
                    fk_injection::ReferenceConvention convention) {
  fk_injection::impl::check_eval_points(points);

  const double p = wave.ray_parameter(model);
  fk_injection::impl::check_supported_slowness(model, p);

  // Largest point depth magnitude, for the float round-off grouping tolerance.
  double max_depth = 0.0;
  const int number_of_points = points.size();
  for (int i = 0; i < number_of_points; ++i) {
    max_depth =
        std::max(max_depth, std::fabs(static_cast<double>(points.z[i])));
  }

  const double tolerance = fk_injection::impl::depth_tolerance(max_depth);
  const fk_injection::impl::DepthGroups groups =
      fk_injection::impl::group_by_depth(points, model, wave, p, tolerance);
  const fk_injection::impl::FkSetup setup = fk_injection::impl::prepare(
      model, wave, window, p, groups, derivative, convention);

  // Delay scaling: 1 in production (exact plane-wave delay); N/(N-1) only when
  // reproducing the SPECFEM3D reference, whose frequency-grid stretched delays.
  const double delay_stretch =
      convention == fk_injection::ReferenceConvention::specfem3d
          ? static_cast<double>(setup.sizes.number_of_fft_samples) /
                static_cast<double>(setup.sizes.number_of_fft_samples - 1)
          : 1.0;

  // Window coverage: every SEM step for every point must map to an abscissa in
  // [0, ncoef-1]; spline evaluation clamps out-of-range values into a silent
  // DC offset otherwise. The guard uses the production (un-stretched) delay:
  // the SPECFEM3D convention may ride the window edge by up to the N/(N-1)
  // stretch and rely on the same spline clamp the reference does.
  const int ncoef = setup.sizes.number_of_stored_frequencies;
  const double resampling = static_cast<double>(setup.sizes.resampling_rate);
  const int npoints = points.size();
  for (int i = 0; i < npoints; ++i) {
    const double offset = fk_injection::impl::point_abscissa_offset(
        setup.abscissa_pad, groups.time_delay[i],
        /*delay_stretch=*/1.0, setup.dt_min);
    const double max_abscissa =
        static_cast<double>(window.nstep - 1) / resampling + offset;
    if (offset < 0.0 || max_abscissa > static_cast<double>(ncoef - 1)) {
      std::ostringstream message;
      message << "solve: sampling abscissa out of range for "
                 "point "
              << i << " (offset " << offset << ", max " << max_abscissa
              << ", coefficients " << ncoef
              << "); increase the injection time-window length";
      throw std::runtime_error(message.str());
    }
  }

  // The series are periodic in the FFT length, so FK times outside one period
  // would silently read another part of the field.
  const double half_period =
      0.5 * static_cast<double>(setup.sizes.number_of_fft_samples);
  const double origin_samples =
      static_cast<double>(wave.origin_time) / setup.dt_min;
  const double first_sample =
      origin_samples - groups.time_delay_max / setup.dt_min;
  const double last_sample =
      origin_samples + static_cast<double>(window.nstep - 1) / resampling -
      groups.time_delay_min / setup.dt_min;
  if (first_sample < -half_period || last_sample >= half_period) {
    std::ostringstream message;
    message << "solve: the run needs FK times from "
            << first_sample * setup.dt_min << " to "
            << last_sample * setup.dt_min << " s, outside the FFT period of "
            << 2.0 * half_period * setup.dt_min
            << " s centered on 0; move origin-time closer to the run or "
               "increase the injection time-window length";
    throw std::runtime_error(message.str());
  }

  const fk_injection::impl::InterfaceStates states =
      fk_injection::impl::solve_layer_stack(model, setup);
  const fk_injection::impl::DepthSeries series =
      fk_injection::impl::depth_series(groups, states, model, setup,
                                       convention);
  return fk_injection::impl::make_result(series, groups, points, setup,
                                         delay_stretch);
}
