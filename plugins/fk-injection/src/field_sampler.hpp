#pragma once

#include "fk_injection/config.hpp"
#include "plane_wave_response.hpp"
#include "series_table.hpp"
#include "specfem/plugin/injection_plugin.hpp"
#include <memory>
#include <span>
#include <vector>

namespace fk_injection {

/// @brief Components per point and step in the output of
/// @c FieldSampler::sample.
inline constexpr int number_of_sample_components = 7;
/// @brief First of 3 components: velocity on elastic points, displacement on
/// acoustic points.
inline constexpr int sample_kinematic_offset = 0;
/// @brief First of 3 components: traction on the point's normal.
inline constexpr int sample_traction_offset = 3;
/// @brief Single component: potential rate (0 on elastic points).
inline constexpr int sample_potential_rate_offset = 6;

/**
 * @brief A point the field is sampled at.
 *
 * Traction uses @f$\sigma_{rr} = \xi_1\sigma_{zz} - 4\xi_\mu d_x@f$ and
 * @f$\sigma_{\theta\theta} = b(\sigma_{rr} + \sigma_{zz})@f$, as SPECFEM3D;
 * with the factors 0 it is @f$\sigma_{rz}@f$ and @f$\sigma_{zz}@f$ on the
 * normal.
 */
struct FieldPoint {
  type_real x = 0;     ///< m
  type_real y = 0;     ///< m
  type_real depth = 0; ///< m, positive down from the model top
  specfem::plugin::Medium medium =
      specfem::plugin::Medium::elastic; ///< Acoustic or elastic
  type_real normal_x = 0;               ///< x-component of the unit normal
  type_real normal_y = 0;               ///< y-component of the unit normal
  type_real normal_z = 0;               ///< z-component of the unit normal
  type_real xi1 = 0;                    ///< First Lame-ratio factor
  type_real xim = 0;                    ///< Shear Lame-ratio factor
  type_real bulk = 0;                   ///< Bulk Lame-ratio factor
};

/**
 * @brief Smallest box holding the points.
 *
 * For callers that see every point, such as tests; ranks of a partitioned mesh
 * must share the mesh box instead.
 *
 * @param points Points
 * @return The box; all zero without points
 */
HorizontalBox bounding_box(const std::vector<FieldPoint> &points);

/** @brief What @c FieldSampler stores for one point. */
struct PointSample {
  int row = 0;       ///< Row of the series table
  double offset = 0; ///< Spline abscissa of step 0
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
 * @brief Samples a series table at a set of points.
 *
 * Each point reads its row at @f$t - \tau(x,y)@f$ and applies the azimuth and
 * traction rotations, so points at one depth share the row.
 */
class FieldSampler {
public:
  /** @brief A sampler without points. */
  FieldSampler() = default;

  /**
   * @brief Construct from per-point data computed elsewhere.
   *
   * @param series          Series table the rows index
   * @param points          Per-point data
   * @param cos_phi         Cosine of the azimuth
   * @param sin_phi         Sine of the azimuth
   * @param resampling_rate SEM steps per resampled sample, >= 1
   * @throws std::runtime_error if @p series is null, a point's row is out of
   *         range or its medium is not acoustic or elastic, or
   *         @p resampling_rate < 1
   */
  FieldSampler(std::shared_ptr<const SeriesTable> series,
               std::vector<PointSample> points, type_real cos_phi,
               type_real sin_phi, int resampling_rate);

  /**
   * @brief Sample @p series at @p points.
   *
   * @param response Response the table was synthesized from
   * @param series   Series table holding a row for every point
   * @param points   Points
   * @throws std::runtime_error if the window doesn't cover a point, which the
   *         response's box check misses for points outside the box, or the
   *         table has no row for one
   */
  FieldSampler(const PlaneWaveResponse &response,
               std::shared_ptr<const SeriesTable> series,
               const std::vector<FieldPoint> &points);

  int number_of_points() const { return static_cast<int>(points_.size()); }
  int resampling_rate() const { return resampling_rate_; }
  const std::vector<PointSample> &points() const { return points_; }
  /// @brief The table sampled; only for a sampler that has one.
  const SeriesTable &series() const { return *series_; }

  /**
   * @brief Reconstruct the field for a window of SEM steps.
   *
   * Fills @p out, row-major (point, component, step-in-window), for SEM steps
   * @c [first_step, first_step + number_of_steps). The @c sample_*_offset
   * constants name the components: kinematic (cos_phi*S0, sin_phi*S0, S1),
   * traction (0 on acoustic points) and potential rate (0 on elastic points).
   *
   * @param first_step      First SEM step in the window
   * @param number_of_steps Window length in SEM steps
   * @param out             Output of size number_of_points *
   *                        number_of_sample_components * number_of_steps
   * @throws std::runtime_error if @p out has any other size
   */
  void sample(int first_step, int number_of_steps,
              std::span<type_real> out) const;

private:
  static std::vector<PointSample>
  point_samples(const PlaneWaveResponse &response, const SeriesTable &series,
                const std::vector<FieldPoint> &points);

  std::shared_ptr<const SeriesTable> series_{};
  std::vector<PointSample> points_{};
  type_real cos_phi_ = 0;
  type_real sin_phi_ = 0;
  int resampling_rate_ = 1;
};

} // namespace fk_injection
