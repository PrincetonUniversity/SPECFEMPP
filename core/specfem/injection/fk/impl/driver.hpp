#pragma once

#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/field_derivative.hpp"
#include "specfem/injection/fk/fk_result.hpp"
#include "specfem/injection/fk/impl/depth_groups.hpp"
#include "specfem/injection/fk/impl/depth_series.hpp"
#include "specfem/injection/fk/impl/fk_setup.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"

namespace specfem {
namespace injection {
namespace fk_impl {

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
 * @return specfem::injection::fk::FkResult ready for device sampling.
 */
specfem::injection::fk::FkResult
make_result(const DepthSeries &series, const DepthGroups &groups,
            const specfem::injection::fk::EvalPoints &points,
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
 * @return specfem::injection::fk::FkResult.
 */
specfem::injection::fk::FkResult
run_fk_driver(const specfem::injection::fk::LayeredModel &model,
              const specfem::injection::fk::IncidentWave &wave,
              const specfem::injection::fk::TimeWindow &window,
              const specfem::injection::fk::EvalPoints &points,
              bool compute_traction,
              specfem::injection::fk::field_derivative derivative,
              bool reproduce_specfem3d_reference = false);

} // namespace fk_impl
} // namespace injection
} // namespace specfem
