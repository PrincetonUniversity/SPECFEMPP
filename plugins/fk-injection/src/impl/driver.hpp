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
 * @brief Reject evaluation points whose arrays disagree.
 *
 * @c x, @c y and @c z must have the same size, @c medium is empty or that size,
 * and the traction inputs (@c normal_* and @c lame_ratio_*) are all empty or
 * all that size: with some of them missing, traction would silently come out
 * zero.
 *
 * @param points Evaluation points.
 * @throws std::runtime_error naming the offending array and its size.
 */
void check_eval_points(const fk_injection::EvalPoints &points);

/**
 * @brief Reject a ray parameter outside the range FK supports.
 *
 * Every vertical slowness has to be real and non-zero, so @p p must be positive
 * and below @f$ 1/v @f$ for every layer velocity @f$ v @f$: @f$ p = 1/v @f$ is
 * 0/0 in the propagators. Since @f$ v_P @f$ of the half-space is among those
 * velocities, incident SV is limited to take-off angles below
 * @f$ rcsin(v_S/v_P) @f$.
 *
 * @param model Layered velocity model.
 * @param p     Horizontal ray parameter (s/m).
 * @throws specfem::plugin::Unsupported if @p p is out of range.
 */
void check_supported_slowness(const fk_injection::LayeredModel &model,
                              double p);

} // namespace fk_injection::impl
