#pragma once

#include "fk_injection/config.hpp"
#include "specfem/plugin/injection_plugin.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace fk_injection {

/// @brief Components per point and step in the output of @c FkResult::sample.
inline constexpr int number_of_sample_components = 7;
/// @brief First of 3 components: displacement or velocity.
inline constexpr int sample_kinematic_offset = 0;
/// @brief First of 3 components: traction on the point's normal.
inline constexpr int sample_traction_offset = 3;
/// @brief Single component: potential rate (0 on elastic points).
inline constexpr int sample_potential_rate_offset = 6;

/**
 * @brief What @c FkResult stores for one evaluation point.
 */
struct PointSample {
  int group = 0;     ///< Depth group index
  double offset = 0; ///< Abscissa offset pad - tau/dt_min
  specfem::plugin::Medium medium =
      specfem::plugin::Medium::elastic; ///< Acoustic or elastic
  type_real xi1 = 0;                    ///< First Lame-ratio factor
  type_real xim = 0;                    ///< Shear Lame-ratio factor
  type_real bulk = 0;                   ///< Bulk Lame-ratio factor
  type_real normal_x = 0;               ///< x-component of the unit normal
  type_real normal_y = 0;               ///< y-component of the unit normal
  type_real normal_z = 0;               ///< z-component of the unit normal
};

/**
 * @brief Depth-grouped FK result with per-point sampling.
 *
 * The FK field factorises as @f$u(x,y,z,t) = U(z, t - \tau(x,y))@f$, so the
 * depth-only series @f$U(z,\cdot)@f$ are stored once per (medium, depth) group
 * as cubic-B-spline coefficients, and each evaluation point carries only its
 * @c PointSample. @c sample reconstructs @f$u(t-\tau)@f$ per point by
 * evaluating the group's splines at the shifted abscissa and applying the
 * azimuth and traction rotations.
 */
class FkResult {
public:
  /** @brief Default-construct an empty, unallocated result. */
  FkResult() = default;

  /**
   * @brief Construct from depth-group series and per-point data.
   *
   * @param series_coefficients Row-major (group, series, coefficient).
   * @param number_of_groups    Number of depth groups.
   * @param coefficient_count   Number of spline coefficients per series.
   * @param points              Per-point data.
   * @param cos_phi, sin_phi    Azimuth rotation (shared by all points).
   * @param resampling_rate     N_P (SEM-step to resampled-grid ratio, >= 1).
   * @throws std::runtime_error if the coefficient array does not match the
   *         group and coefficient counts, a point's group is out of range or
   *         its medium is not acoustic or elastic, or @p resampling_rate < 1.
   */
  FkResult(std::vector<type_real> series_coefficients, int number_of_groups,
           int coefficient_count, std::vector<PointSample> points,
           type_real cos_phi, type_real sin_phi, int resampling_rate);

  int number_of_points() const { return static_cast<int>(points_.size()); }
  int number_of_groups() const { return number_of_groups_; }
  int coefficient_count() const { return coefficient_count_; }
  int resampling_rate() const { return resampling_rate_; }

  const std::vector<PointSample> &points() const { return points_; }

  /**
   * @brief Reconstruct the field for a window of SEM steps.
   *
   * Fills @p out, row-major (point, component, step-in-window), for SEM steps
   * @c [first_step, first_step + number_of_steps). The @c sample_*_offset
   * constants name the components: kinematic (cos_phi*S0, sin_phi*S0, S1),
   * traction (0 on acoustic points) and potential rate (0 on elastic points).
   *
   * @param first_step      First SEM step in the window.
   * @param number_of_steps Window length in SEM steps.
   * @param out             Output of size number_of_points *
   *                        number_of_sample_components * number_of_steps.
   * @throws std::runtime_error if @p out has any other size.
   */
  void sample(int first_step, int number_of_steps,
              std::span<type_real> out) const;

private:
  std::size_t coefficient_index(int group, int series, int coefficient) const;

  std::vector<type_real> series_coefficients_{};
  std::vector<PointSample> points_{};
  type_real cos_phi_ = 0;
  type_real sin_phi_ = 0;
  int resampling_rate_ = 0;
  int number_of_groups_ = 0;
  int coefficient_count_ = 0;
};

} // namespace fk_injection
