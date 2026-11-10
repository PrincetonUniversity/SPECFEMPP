#pragma once

#include "fk_injection/config.hpp"
#include "media.hpp"
#include "plane_wave_response.hpp"
#include "series_table.hpp"
#include "specfem/plugin/injection_plugin.hpp"
#include <memory>
#include <span>
#include <vector>

namespace fk_injection {

/** @brief A point the field is sampled at. */
struct FieldPoint {
  type_real x = 0;        ///< m
  type_real y = 0;        ///< m
  type_real depth = 0;    ///< m, positive down from the model top
  int layer = 0;          ///< Index into @c LayeredModel::layers
  type_real normal_x = 0; ///< x-component of the unit normal
  type_real normal_y = 0; ///< y-component of the unit normal
  type_real normal_z = 0; ///< z-component of the unit normal
};

/**
 * @brief Samples a series table at a set of points in one medium.
 *
 * Each point reads its row at @f$t - \tau(x,y)@f$ and forms the requested
 * quantity with @c MediumType::project, so points at one depth share the row.
 *
 * @tparam MediumType @c ElasticMedium or @c AcousticMedium
 */
template <typename MediumType> class FieldSampler {
public:
  using Point = typename MediumType::Point;

  /**
   * @brief Construct from per-point data computed elsewhere.
   *
   * @param series          Series table the rows index
   * @param points          Per-point data
   * @param cos_phi         Cosine of the azimuth
   * @param sin_phi         Sine of the azimuth
   * @param resampling_rate SEM steps per resampled sample, >= 1
   * @throws std::runtime_error if @p series is null, a point's row is out of
   *         range or of another medium, or @p resampling_rate < 1
   */
  FieldSampler(std::shared_ptr<const SeriesTable> series,
               std::vector<Point> points, type_real cos_phi, type_real sin_phi,
               int resampling_rate);

  /**
   * @brief Sample @p series at @p points.
   *
   * The traction factors are those of the point's model layer.
   *
   * @param response Response the table was synthesized from
   * @param series   Series table holding a row for every point
   * @param points   Points
   * @throws std::runtime_error if the window doesn't cover a point, which the
   *         response's box check misses for points outside the box, or the
   *         table has no row of this medium for one
   */
  FieldSampler(const PlaneWaveResponse &response,
               std::shared_ptr<const SeriesTable> series,
               const std::vector<FieldPoint> &points);

  int number_of_points() const { return static_cast<int>(points_.size()); }
  int resampling_rate() const { return resampling_rate_; }
  const std::vector<Point> &points() const { return points_; }
  const SeriesTable &series() const { return *series_; }

  /**
   * @brief Reconstruct one quantity for a window of SEM steps.
   *
   * Fills @p out, row-major (point, component, step-in-window), for SEM steps
   * @c [first_step, first_step + number_of_steps), evaluating only the series
   * @p quantity needs.
   *
   * @param quantity        One of @c MediumType::quantities
   * @param first_step      First SEM step in the window
   * @param number_of_steps Window length in SEM steps
   * @param out             Output of size number_of_points *
   *                        number_of_components(quantity) * number_of_steps
   * @throws std::runtime_error if the medium doesn't serve @p quantity or
   *         @p out has any other size
   */
  void sample(specfem::plugin::Quantity quantity, int first_step,
              int number_of_steps, std::span<type_real> out) const;

private:
  static std::vector<Point>
  point_samples(const PlaneWaveResponse &response, const SeriesTable &series,
                const std::vector<FieldPoint> &points);

  std::shared_ptr<const SeriesTable> series_{};
  std::vector<Point> points_{};
  type_real cos_phi_ = 0;
  type_real sin_phi_ = 0;
  int resampling_rate_ = 1;
};

} // namespace fk_injection
