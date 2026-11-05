#pragma once

#include "field_sampler.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/reference_convention.hpp"
#include "fk_injection/time_window.hpp"
#include "plane_wave_response.hpp"
#include "series_table.hpp"

#include <memory>
#include <vector>

namespace fk_injection_tests_impl {

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
  return field_at(model, wave, window, fk_injection::bounding_box(points),
                  points, convention);
}

} // namespace fk_injection_tests_impl
