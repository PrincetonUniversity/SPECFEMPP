#include "impl/driver.hpp"

#include "impl/layer_stack.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
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

  std::vector<int> point_group(npoints);
  // Per-point abscissa offset pad - tau/dt_min.
  std::vector<double> point_offset(npoints);
  // Per-point traction material: xi1, xim, bulk, nx, ny, nz.
  std::vector<type_real> point_material(static_cast<std::size_t>(npoints) * 6,
                                        0);
  std::vector<int> point_is_acoustic(npoints);

  const bool have_lame = !points.lame_ratio_xi1.empty();
  const bool have_normal = !points.normal_x.empty();

  const double dt_min = setup.dt_min;
  const double pad = static_cast<double>(setup.pad_samples);

  for (int i = 0; i < npoints; ++i) {
    const int g = groups.point_group[i];
    point_group[i] = g;
    point_offset[i] = fk_injection::impl::point_abscissa_offset(
        pad, groups.time_delay[i], delay_stretch, dt_min);
    point_is_acoustic[i] =
        groups.groups[g].medium == fk_injection::point_medium::acoustic ? 1 : 0;
    if (have_lame) {
      point_material[6 * i + 0] = points.lame_ratio_xi1[i];
      point_material[6 * i + 1] = points.lame_ratio_xim[i];
      point_material[6 * i + 2] = points.lame_ratio_bulk[i];
    }
    if (have_normal) {
      point_material[6 * i + 3] = points.normal_x[i];
      point_material[6 * i + 4] = points.normal_y[i];
      point_material[6 * i + 5] = points.normal_z[i];
    }
  }

  const type_real cos_phi = static_cast<type_real>(std::cos(setup.phi));
  const type_real sin_phi = static_cast<type_real>(std::sin(setup.phi));

  return fk_injection::FkResult(coefficients, point_group, point_offset,
                                point_material, point_is_acoustic, cos_phi,
                                sin_phi, setup.sizes.resampling_rate,
                                setup.compute_traction);
}

fk_injection::FkResult
fk_injection::impl::run_fk_driver(const fk_injection::LayeredModel &model,
                                  const fk_injection::IncidentWave &wave,
                                  const fk_injection::TimeWindow &window,
                                  const fk_injection::EvalPoints &points,
                                  bool compute_traction,
                                  fk_injection::field_derivative derivative,
                                  bool reproduce_specfem3d_reference) {
  const double p = fk_injection::impl::ray_parameter(model, wave);

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
