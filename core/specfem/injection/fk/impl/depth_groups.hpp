#pragma once

#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include <vector>

namespace specfem {
namespace injection {
namespace fk_impl {

/**
 * @brief One (medium, depth) group of evaluation points.
 *
 * All points in a group share the same depth-only field series; they differ
 * only by a per-point time delay and linear weights. The geometry mirrors
 * @c LayeredModel::LayerLocation.
 */
struct DepthGroup {
  specfem::injection::fk::point_medium medium; ///< resolved medium
  int layer;                        ///< index within the fluid or elastic block
  double height;                    ///< height above the bottom of @c layer (m)
  double depth_below_halfspace_top; ///< zz, half-space top minus depth (m)
  bool in_halfspace; ///< true when the group sits in the half-space
  bool free_surface; ///< true at the top of the fluid column
  double depth;      ///< representative (anchor) depth of the group
};

/**
 * @brief Depth grouping of a point set plus per-point delays.
 *
 * @c point_group[i] is the group index of point i; @c time_delay[i] is its
 * delay tau in seconds. @c time_delay_min / @c time_delay_max bound the delay
 * span (used to size the stored window so it covers t - tau for every point).
 */
struct DepthGroups {
  std::vector<DepthGroup> groups;
  std::vector<int> point_group;
  std::vector<double> time_delay;
  double time_delay_min = 0.0;
  double time_delay_max = 0.0;
};

/**
 * @brief Horizontal ray parameter p = sin(theta) / V_halfspace.
 *
 * Uses the half-space P velocity for an incident P wave and the half-space S
 * velocity for an incident SV wave.
 *
 * @param model Layered velocity model (its last elastic layer is the
 * halfspace).
 * @param wave  Incident plane wave (type and take-off angle).
 * @return Horizontal slowness p in s/m.
 */
double ray_parameter(const specfem::injection::fk::LayeredModel &model,
                     const specfem::injection::fk::IncidentWave &wave);

/**
 * @brief Floating-point round-off slack for exact-depth grouping, in metres.
 *
 * This is NOT a physical tolerance: grouping is mesh-driven, and a structured
 * mesh places every same-level GLL point at an identical depth. The returned
 * value is only the single-precision round-off at @p max_depth
 * (@f$ 8\,\text{ulp}(z_{max}) @f$), so points the mesh placed at the same depth
 * group together while physically-distinct depth levels never merge. There is
 * no dependence on any absolute physical distance (velocity, time step).
 *
 * @param max_depth Largest evaluation-point depth magnitude (m).
 * @return Round-off slack in metres.
 */
double depth_tolerance(double max_depth);

/**
 * @brief Group evaluation points by (medium, exact depth) and compute per-point
 *        tau.
 *
 * Resolves each point's medium, sorts point indices by (medium, depth), and
 * groups consecutive points at the SAME depth (equal to within @p tolerance,
 * which is float round-off only — see @c depth_tolerance). Grouping is thus
 * mesh-exact: a structured mesh's same-level points share one group, and
 * distinct depth levels are never merged, so each group's field is computed at
 * its exact depth with no approximation. Each group's geometry comes from
 * @c LayeredModel::locate at the group depth. The per-point delay is
 * @f$ \tau = p[(x-x_0)\cos\phi + (y-y_0)\sin\phi] + \eta_p(0 - z_0) @f$,
 * @f$ \eta_p = \sqrt{\max(0, 1/v_{P,hs}^2 - p^2)} @f$ (the half-space P
 * slowness, matching the reference for both wave types).
 *
 * @note Correct for any boundary shape (each distinct depth is synthesized
 *       exactly). Reuse is full on a Cartesian/axis-aligned boundary; a
 *       diagonally-sloped or curved boundary yields mostly singleton groups, so
 *       the shared synthesis is lost (still correct, still memory-light). The
 *       deferred extension for efficient curved-boundary injection is to
 *       interpolate the series along depth on a band-limited reference grid
 *       (the depth analogue of the cubic time-spline), localized to this stage
 *       and @c FkResult::sample.
 *
 * @param points    Evaluation points (host-resident coordinates and optional
 *                  per-point medium overrides).
 * @param model     Layered velocity model.
 * @param wave      Incident plane wave (origin, azimuth, type).
 * @param p         Horizontal ray parameter from @c ray_parameter.
 * @param tolerance Float round-off slack from @c depth_tolerance (not
 * physical).
 * @return Populated DepthGroups.
 * @throws std::runtime_error if a point's medium cannot be resolved
 *         (propagated from @c LayeredModel::locate).
 */
DepthGroups group_by_depth(const specfem::injection::fk::EvalPoints &points,
                           const specfem::injection::fk::LayeredModel &model,
                           const specfem::injection::fk::IncidentWave &wave,
                           double p, double tolerance);

} // namespace fk_impl
} // namespace injection
} // namespace specfem
