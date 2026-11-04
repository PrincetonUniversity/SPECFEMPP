#pragma once

#include "fk_injection/eval_points.hpp"
#include "fk_injection/field_derivative.hpp"
#include "fk_injection/fk_result.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/time_window.hpp"
#include "impl/depth_groups.hpp"
#include "impl/depth_series.hpp"
#include "impl/fk_setup.hpp"

namespace fk_injection::impl {

/**
 * @brief Spline abscissa offset for one point: @c pad -
 * tau*delay_stretch/dt_min.
 *
 * Single source of the per-point offset so the window-coverage check and the
 * stored @c point_offset cannot drift apart.
 *
 * @param pad_samples  Leading pad in resampled-grid samples.
 * @param time_delay   Per-point plane-wave delay tau in s.
 * @param delay_stretch Delay scaling (1 in production; N/(N-1) for the
 *                      SPECFEM3D-reference reproduction).
 * @param dt_min       Resampled-grid time step in s.
 * @return Abscissa offset in resampled-grid samples.
 */
inline double point_abscissa_offset(double pad_samples, double time_delay,
                                    double delay_stretch, double dt_min) {
  return pad_samples - time_delay * delay_stretch / dt_min;
}

/**
 * @brief Assemble the device-resident grouped result.
 *
 * Copies the depth-series coefficients to device @c type_real, and builds the
 * per-point data: group index, abscissa offset (see @c point_abscissa_offset),
 * the traction material factors (xi1, xim, bulk, normal), and the acoustic
 * flag.
 *
 * @param series Depth-series coefficients + t0/pad bookkeeping.
 * @param groups Depth groups (point->group map, delays).
 * @param points Evaluation points (Lame factors, normals).
 * @param setup  Shared setup (azimuth, dt_min, resampling, traction flag).
 * @param delay_stretch Delay scaling applied to every point's offset (1 in
 *        production; N/(N-1) for the SPECFEM3D-reference reproduction).
 * @return fk_injection::FkResult ready for device sampling.
 */
fk_injection::FkResult make_result(const DepthSeries &series,
                                   const DepthGroups &groups,
                                   const fk_injection::EvalPoints &points,
                                   const FkSetup &setup, double delay_stretch);

/**
 * @brief Depth-grouped FK driver: host pipeline + one device sampling kernel.
 *
 * Runs ray_parameter -> depth_tolerance -> group_by_depth -> prepare ->
 * solve_layer_stack -> depth_series -> make_result. Throws if the sampling
 * window does not cover every SEM step and delay.
 *
 * @param model            Layered velocity model.
 * @param wave             Incident plane wave.
 * @param window           Time-window parameters.
 * @param points           Evaluation points.
 * @param compute_traction Whether to produce traction series.
 * @param derivative       Kinematic quantity for the vector components.
 * @param reproduce_specfem3d_reference Validation only (see @c make_result);
 *        default false = physically-exact delay.
 * @return fk_injection::FkResult.
 */
fk_injection::FkResult
run_fk_driver(const fk_injection::LayeredModel &model,
              const fk_injection::IncidentWave &wave,
              const fk_injection::TimeWindow &window,
              const fk_injection::EvalPoints &points, bool compute_traction,
              fk_injection::field_derivative derivative,
              bool reproduce_specfem3d_reference = false);

} // namespace fk_injection::impl
