#include "impl/driver.hpp"

#include "impl/layer_stack.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

fk_injection::FkResult
fk_injection::impl::make_result(const fk_injection::impl::DepthSeries &series,
                                const fk_injection::impl::DepthGroups &groups,
                                const fk_injection::EvalPoints &points,
                                const fk_injection::impl::FkSetup &setup,
                                double delay_stretch) {
  const int npoints = points.size();

  // Series coefficients: double -> type_real, same layout.
  std::vector<type_real> coefficients(series.coefficients.begin(),
                                      series.coefficients.end());

  const bool have_lame = !points.lame_ratio_xi1.empty();
  const bool have_normal = !points.normal_x.empty();

  std::vector<fk_injection::PointSample> samples(npoints);
  for (int i = 0; i < npoints; ++i) {
    fk_injection::PointSample &sample = samples[i];
    sample.group = groups.point_group[i];
    sample.offset = fk_injection::impl::point_abscissa_offset(
        static_cast<double>(setup.pad_samples), groups.time_delay[i],
        delay_stretch, setup.dt_min);
    sample.medium = groups.groups[sample.group].medium;
    if (have_lame) {
      sample.xi1 = points.lame_ratio_xi1[i];
      sample.xim = points.lame_ratio_xim[i];
      sample.bulk = points.lame_ratio_bulk[i];
    }
    if (have_normal) {
      sample.normal_x = points.normal_x[i];
      sample.normal_y = points.normal_y[i];
      sample.normal_z = points.normal_z[i];
    }
  }

  return fk_injection::FkResult(
      std::move(coefficients), series.number_of_groups,
      series.coefficient_count, std::move(samples),
      static_cast<type_real>(std::cos(setup.phi)),
      static_cast<type_real>(std::sin(setup.phi)), setup.sizes.resampling_rate);
}

fk_injection::FkResult
fk_injection::impl::run_fk_driver(const fk_injection::LayeredModel &model,
                                  const fk_injection::IncidentWave &wave,
                                  const fk_injection::TimeWindow &window,
                                  const fk_injection::EvalPoints &points,
                                  bool compute_traction,
                                  fk_injection::field_derivative derivative,
                                  bool reproduce_specfem3d_reference) {
  const double p = wave.ray_parameter(model);

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
      model, wave, window, p, groups, derivative, compute_traction);

  // Delay scaling: 1 in production (exact plane-wave delay); N/(N-1) only when
  // reproducing the SPECFEM3D reference, whose frequency-grid stretched delays.
  const double delay_stretch =
      reproduce_specfem3d_reference
          ? static_cast<double>(setup.sizes.number_of_fft_samples) /
                static_cast<double>(setup.sizes.number_of_fft_samples - 1)
          : 1.0;

  // Window coverage: every SEM step for every point must map to an abscissa in
  // [0, ncoef-1]; spline evaluation clamps out-of-range values into a silent
  // DC offset otherwise. The guard uses the production (un-stretched) delay:
  // the SPECFEM3D-reference mode may ride the window edge by up to the N/(N-1)
  // stretch and rely on the same spline clamp the reference does.
  const int ncoef = setup.sizes.number_of_stored_frequencies;
  const double resampling = static_cast<double>(setup.sizes.resampling_rate);
  const int npoints = points.size();
  for (int i = 0; i < npoints; ++i) {
    const double offset = fk_injection::impl::point_abscissa_offset(
        static_cast<double>(setup.pad_samples), groups.time_delay[i],
        /*delay_stretch=*/1.0, setup.dt_min);
    const double max_abscissa =
        static_cast<double>(window.nstep - 1) / resampling + offset;
    if (offset < 0.0 || max_abscissa > static_cast<double>(ncoef - 1)) {
      std::ostringstream message;
      message << "run_fk_driver: sampling abscissa out of range for "
                 "point "
              << i << " (offset " << offset << ", max " << max_abscissa
              << ", coefficients " << ncoef
              << "); increase the injection time-window length";
      throw std::runtime_error(message.str());
    }
  }

  const fk_injection::impl::InterfaceStates states =
      fk_injection::impl::solve_layer_stack(model, setup);
  const fk_injection::impl::DepthSeries series =
      fk_injection::impl::depth_series(groups, states, model, setup,
                                       reproduce_specfem3d_reference);
  return fk_injection::impl::make_result(series, groups, points, setup,
                                         delay_stretch);
}
