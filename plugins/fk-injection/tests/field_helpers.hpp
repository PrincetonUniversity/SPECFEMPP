#pragma once

#include "field_sampler.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/reference_convention.hpp"
#include "fk_injection/time_window.hpp"
#include "plane_wave_response.hpp"
#include "series_table.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace fk_injection_tests_impl {

/**
 * @brief Smallest box holding the points; all zero without points.
 *
 * Tests see every point; ranks of a partitioned mesh share the mesh box.
 */
inline fk_injection::HorizontalBox
bounding_box(const std::vector<fk_injection::FieldPoint> &points) {
  if (points.empty())
    return {};
  fk_injection::HorizontalBox box{ static_cast<double>(points[0].x),
                                   static_cast<double>(points[0].x),
                                   static_cast<double>(points[0].y),
                                   static_cast<double>(points[0].y) };
  for (const auto &point : points) {
    box.x_min = std::min(box.x_min, static_cast<double>(point.x));
    box.x_max = std::max(box.x_max, static_cast<double>(point.x));
    box.y_min = std::min(box.y_min, static_cast<double>(point.y));
    box.y_max = std::max(box.y_max, static_cast<double>(point.y));
  }
  return box;
}

/** @brief The FK field at the points, sized from @p box. */
inline fk_injection::FieldSampler
field_at(const fk_injection::LayeredModel &model,
         const fk_injection::IncidentWave &wave,
         const fk_injection::TimeWindow &window,
         const fk_injection::HorizontalBox &box,
         const std::vector<fk_injection::FieldPoint> &points,
         const fk_injection::ReferenceConvention convention =
             fk_injection::ReferenceConvention::exact) {
  const fk_injection::PlaneWaveResponse response(model, wave, window, box,
                                                 convention);
  std::vector<fk_injection::SeriesKey> keys;
  for (const auto &point : points)
    keys.push_back({ point.medium, point.depth });
  return fk_injection::FieldSampler(
      response,
      std::make_shared<const fk_injection::SeriesTable>(
          fk_injection::SeriesTable::synthesize(response, keys)),
      points);
}

/** @brief The FK field at the points, sized from their bounding box. */
inline fk_injection::FieldSampler
field_at(const fk_injection::LayeredModel &model,
         const fk_injection::IncidentWave &wave,
         const fk_injection::TimeWindow &window,
         const std::vector<fk_injection::FieldPoint> &points,
         const fk_injection::ReferenceConvention convention =
             fk_injection::ReferenceConvention::exact) {
  return field_at(model, wave, window, bounding_box(points), points,
                  convention);
}

} // namespace fk_injection_tests_impl
