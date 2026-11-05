#pragma once

#include "fk_injection/eval_points.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/time_window.hpp"

namespace fk_injection::impl {

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
