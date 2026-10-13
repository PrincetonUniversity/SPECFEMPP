#pragma once

#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/injection_points.hpp"

namespace specfem {
namespace injection {

/**
 * @brief Internal helpers for the FK boundary coupling.
 */
namespace fk_coupling_impl {

/**
 * @brief Resolve a point's FK medium and assert mesh-vs-model consistency.
 *
 * Maps the generic medium tag of point @p i to an FK @c point_medium and checks
 * that the point's local material @c (rho, vp, vs) matches the 1-D @p model
 * layer at the point's depth within @p tolerance.
 *
 * @param points Method-agnostic injection points.
 * @param i Point index.
 * @param model 1-D layered model.
 * @param tolerance Relative material tolerance.
 * @return The resolved FK medium for point @p i.
 * @throws std::runtime_error on an unsupported medium or a material mismatch.
 */
specfem::injection::fk::point_medium resolve_medium_and_check(
    const specfem::injection::injection_points &points, int i,
    const specfem::injection::fk::LayeredModel &model, double tolerance);

} // namespace fk_coupling_impl

namespace fk {

/**
 * @brief Convert method-agnostic injection points into FK evaluation points.
 *
 * This is where the FK method owns its traction derivation: it maps the generic
 * medium tag to @c point_medium and derives the Lamé ratios
 * (\f$\xi_1 = v_p^2/v_s^2\f$, \f$\xi_\mu = 1/2\f$,
 * \f$\mathrm{bulk} = \lambda/(2(\lambda+\mu))\f$) from the per-point
 * \f$(v_p, v_s)\f$ carried by @p points. It also asserts that the local mesh
 * material matches the 1-D @p model at each point's depth — the incident field
 * is defined by @p model, so a mismatch means the injected wavefield would be
 * unrelated to the local medium.
 *
 * @param points Method-agnostic points (geometry + medium + local material).
 * @param model 1-D layered velocity model driving the FK synthesis.
 * @param compute_traction When true, fill normals and Lamé ratios (needed for
 *        traction); otherwise leave those Views empty.
 * @param tolerance Relative tolerance for the model-vs-mesh material check.
 * @return FK @c EvalPoints ready for @c solve.
 * @throws std::runtime_error on an unsupported medium or a material mismatch
 *         beyond @p tolerance (message names the point, depth, and both
 *         material triplets).
 */
EvalPoints
eval_points_from_points(const specfem::injection::injection_points &points,
                        const LayeredModel &model, bool compute_traction,
                        double tolerance);

} // namespace fk
} // namespace injection
} // namespace specfem
