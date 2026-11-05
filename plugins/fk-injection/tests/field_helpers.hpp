#pragma once

#include "field_sampler.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/reference_convention.hpp"
#include "fk_injection/time_window.hpp"
#include "plane_wave_response.hpp"
#include "series_table.hpp"

#include <algorithm>
#include <cmath>
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

/** @brief Depth tolerance of a table serving the points. */
inline double
depth_tolerance(const std::vector<fk_injection::FieldPoint> &points) {
  double deepest = 0.0;
  for (const auto &point : points)
    deepest = std::max(deepest, std::fabs(static_cast<double>(point.depth)));
  return fk_injection::SeriesTable::depth_tolerance(deepest);
}

/**
 * @brief Keys in the layers @c LayeredModel::locate assigns to the depths in
 *        @p medium.
 */
inline std::vector<fk_injection::SeriesKey>
keys_at(const fk_injection::LayeredModel &model,
        const specfem::plugin::Medium medium,
        const std::vector<double> &depths) {
  std::vector<fk_injection::SeriesKey> keys;
  for (const double depth : depths)
    keys.push_back(
        { model
              .locate(depth, medium,
                      fk_injection::SeriesTable::depth_tolerance(depth))
              .layer,
          static_cast<fk_injection::type_real>(depth) });
  return keys;
}

/** @brief The table for the keys, with the tolerance of the deepest. */
inline fk_injection::SeriesTable
table_for(const fk_injection::PlaneWaveResponse &response,
          const std::vector<fk_injection::SeriesKey> &keys) {
  double deepest = 0.0;
  for (const auto &key : keys)
    deepest = std::max(deepest, std::fabs(static_cast<double>(key.depth)));
  return fk_injection::SeriesTable::synthesize(
      response, keys, fk_injection::SeriesTable::depth_tolerance(deepest));
}

/**
 * @brief The FK field at the points, sized from @p box.
 *
 * Each point goes in the layer @c LayeredModel::locate assigns to its depth in
 * its medium of @p media (elastic when empty), replacing its @c layer.
 */
inline fk_injection::FieldSampler
field_at(const fk_injection::LayeredModel &model,
         const fk_injection::IncidentWave &wave,
         const fk_injection::TimeWindow &window,
         const fk_injection::HorizontalBox &box,
         std::vector<fk_injection::FieldPoint> points,
         const fk_injection::ReferenceConvention convention =
             fk_injection::ReferenceConvention::exact,
         const std::vector<specfem::plugin::Medium> &media = {}) {
  const fk_injection::PlaneWaveResponse response(model, wave, window, box,
                                                 convention);
  const double tolerance = depth_tolerance(points);
  std::vector<fk_injection::SeriesKey> keys;
  for (std::size_t i = 0; i < points.size(); ++i) {
    auto &point = points[i];
    point.layer =
        model
            .locate(static_cast<double>(point.depth),
                    media.empty() ? specfem::plugin::Medium::elastic : media[i],
                    tolerance)
            .layer;
    keys.push_back({ point.layer, point.depth });
  }
  return fk_injection::FieldSampler(
      response,
      std::make_shared<const fk_injection::SeriesTable>(
          fk_injection::SeriesTable::synthesize(response, keys, tolerance)),
      points);
}

/** @brief The FK field at the points, sized from their bounding box. */
inline fk_injection::FieldSampler
field_at(const fk_injection::LayeredModel &model,
         const fk_injection::IncidentWave &wave,
         const fk_injection::TimeWindow &window,
         const std::vector<fk_injection::FieldPoint> &points,
         const fk_injection::ReferenceConvention convention =
             fk_injection::ReferenceConvention::exact,
         const std::vector<specfem::plugin::Medium> &media = {}) {
  return field_at(model, wave, window, bounding_box(points), points, convention,
                  media);
}

} // namespace fk_injection_tests_impl
