#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/utilities/cubic_bspline.hpp"
#include <cstddef>
#include <vector>

namespace fk_injection {

/**
 * @brief Depth-grouped FK result with per-point sampling.
 *
 * The FK field factorises as @f$u(x,y,z,t) = U(z, t - \tau(x,y))@f$, so the
 * depth-only series @f$U(z,\cdot)@f$ are stored once per (medium, depth) group
 * as cubic-B-spline coefficients, and each evaluation point carries only its
 * group index, an abscissa offset @f$\text{pad} - \tau/\Delta t_\text{min}@f$,
 * the traction material factors, and a medium flag. @c sample reconstructs
 * @f$u(t-\tau)@f$ per point by evaluating the group's splines at the shifted
 * abscissa and applying the azimuth and traction rotations.
 *
 * Built by @c fk_injection::impl::make_result; consumed by the plugin adapter.
 */
class FkResult {
public:
  /** @brief Default-construct an empty, unallocated result. */
  FkResult() = default;

  /**
   * @brief Construct from row-major arrays.
   *
   * @param series_coefficients (group, series, coefficient).
   * @param number_of_groups    Number of depth groups.
   * @param coefficient_count   Number of spline coefficients per series.
   * @param point_group         (point) group index.
   * @param point_offset        (point) abscissa offset pad - tau/dt_min.
   * @param point_material      (point, 6) xi1, xim, bulk, nx, ny, nz.
   * @param point_is_acoustic   (point) 1 for acoustic, 0 for elastic.
   * @param cos_phi, sin_phi    Azimuth rotation (shared by all points).
   * @param resampling_rate     N_P (SEM-step to resampled-grid ratio).
   * @param has_traction        Whether traction components are meaningful.
   */
  FkResult(std::vector<type_real> series_coefficients, int number_of_groups,
           int coefficient_count, std::vector<int> point_group,
           std::vector<double> point_offset,
           std::vector<type_real> point_material,
           std::vector<int> point_is_acoustic, type_real cos_phi,
           type_real sin_phi, int resampling_rate, bool has_traction);

  int number_of_points() const { return number_of_points_; }
  int number_of_groups() const { return number_of_groups_; }
  int coefficient_count() const { return coefficient_count_; }
  int resampling_rate() const { return resampling_rate_; }
  bool has_traction() const { return has_traction_; }
  type_real cos_phi() const { return cos_phi_; }
  type_real sin_phi() const { return sin_phi_; }

  /**
   * @brief One stored spline coefficient.
   *
   * @param group       Depth group.
   * @param series      Series slot.
   * @param coefficient Coefficient within the series.
   * @return The coefficient.
   */
  type_real series_coefficient(int group, int series, int coefficient) const;

  const std::vector<int> &point_group() const { return point_group_; }
  const std::vector<double> &point_offset() const { return point_offset_; }

  /**
   * @brief Reconstruct the field for a window of SEM steps.
   *
   * Fills @p out, row-major (point, component, step-in-window), for SEM steps
   * @c [first_step, first_step + number_of_steps). Components:
   * 0..2 displacement/velocity (cos_phi*S0, sin_phi*S0, S1); 3..5 traction
   * (when @p number_of_components >= 6); 6 pressure (when
   * @p number_of_components >= 7).
   *
   * @param first_step           First SEM step in the window.
   * @param number_of_steps      Window length in SEM steps.
   * @param number_of_components Components per point and step in @p out.
   * @param out                  Output of size number_of_points *
   *                             number_of_components * number_of_steps.
   */
  void sample(int first_step, int number_of_steps, int number_of_components,
              type_real *out) const;

private:
  std::size_t coefficient_index(int group, int series, int coefficient) const;

  std::vector<type_real> series_coefficients_{};
  std::vector<int> point_group_{};
  std::vector<double> point_offset_{};
  std::vector<type_real> point_material_{};
  std::vector<int> point_is_acoustic_{};
  type_real cos_phi_ = 0;
  type_real sin_phi_ = 0;
  int resampling_rate_ = 0;
  int number_of_points_ = 0;
  int number_of_groups_ = 0;
  int coefficient_count_ = 0;
  bool has_traction_ = false;
};

} // namespace fk_injection
