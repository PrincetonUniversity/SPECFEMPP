#pragma once

#include "fk_injection/eval_points.hpp"
#include "fk_injection/field_derivative.hpp"
#include "fk_injection/fk_result.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/time_window.hpp"

namespace fk_injection {

/**
 * @brief How @c solve treats the two points where it deliberately differs from
 *        SPECFEM3D's FK routine.
 *
 * - @c exact : the inverse FFT normalizes by the resampled step @c dt_min, the
 *   frequency grid is @f$ 1/(N \Delta t_\text{min}) @f$, and each point's delay
 *   is the exact plane-wave delay.
 * - @c specfem3d : reproduces SPECFEM3D for comparison against it. The inverse
 *   FFT normalizes by the SEM step, so amplitudes are @f$ N_P @f$ times larger,
 *   and its frequency grid stretches all times by @f$ N/(N-1) @f$.
 */
enum class ReferenceConvention { exact, specfem3d };

/**
 * @brief Compute the FK field at a set of evaluation points.
 *
 * Runs on the host: solves the layer stack once per frequency, synthesizes one
 * set of depth-only series per (medium, depth) group, and stores them as
 * cubic-B-spline coefficients for @c FkResult::sample.
 *
 * @param model      1-D horizontally-layered velocity model.
 * @param wave       Incident plane-wave descriptor (type, angles, origin,
 *                   amplitude, Gaussian half-duration).
 * @param window     Simulation time-step and windowing parameters used to
 *                   derive FK working-array sizes.
 * @param points     Evaluation-point coordinates, with optional normals and
 *                   Lame factors. Traction is zero without them.
 * @param derivative Kinematic quantity stored in the three vector components.
 * @param convention See @c ReferenceConvention.
 * @return FkResult holding the kinematic field, traction and potential rate.
 * @throws specfem::plugin::Unsupported if the ray parameter is out of range
 *         (see @c impl::check_supported_slowness).
 * @throws std::runtime_error if the point arrays disagree, a point can't be
 *         located, or the time window doesn't cover every step and delay.
 */
FkResult solve(const LayeredModel &model, const IncidentWave &wave,
               const TimeWindow &window, const EvalPoints &points,
               field_derivative derivative,
               ReferenceConvention convention = ReferenceConvention::exact);

} // namespace fk_injection
