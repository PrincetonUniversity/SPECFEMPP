#pragma once

#include "field_sampler.hpp"
#include "fk_injection/eval_points.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/reference_convention.hpp"
#include "fk_injection/time_window.hpp"

namespace fk_injection {

/**
 * @brief Compute the FK field at a set of evaluation points.
 *
 * Runs on the host: solves the layer stack once per frequency, synthesizes one
 * set of depth-only series per (medium, depth) group, and stores them as
 * cubic-B-spline coefficients for @c FieldSampler::sample.
 *
 * @param model      1-D horizontally-layered velocity model.
 * @param wave       Incident plane-wave descriptor (type, angles, origin,
 *                   amplitude, Gaussian half-duration).
 * @param window     Simulation time-step and windowing parameters used to
 *                   derive FK working-array sizes.
 * @param points     Evaluation-point coordinates, with optional normals and
 *                   Lame factors. Traction is zero without them.
 * @param convention See @c ReferenceConvention.
 * @return Sampler of velocity on elastic points, displacement on acoustic
 *         points, traction and potential rate. A point without a medium takes
 *         the one of its layer.
 * @throws specfem::plugin::Unsupported if the ray parameter is out of range
 *         (see @c impl::check_supported_slowness).
 * @throws std::runtime_error if the point arrays disagree, a point can't be
 *         located, or the time window doesn't cover every step and delay.
 */
FieldSampler solve(const LayeredModel &model, const IncidentWave &wave,
                   const TimeWindow &window, const EvalPoints &points,
                   ReferenceConvention convention = ReferenceConvention::exact);

} // namespace fk_injection
