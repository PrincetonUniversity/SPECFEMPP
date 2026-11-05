#include "fk_injection/solver.hpp"
#include "impl/driver.hpp"
#include "plane_wave_response.hpp"
#include "series_table.hpp"
#include <algorithm>
#include <memory>
#include <optional>

fk_injection::FieldSampler
fk_injection::solve(const fk_injection::LayeredModel &model,
                    const fk_injection::IncidentWave &wave,
                    const fk_injection::TimeWindow &window,
                    const fk_injection::EvalPoints &points,
                    fk_injection::ReferenceConvention convention) {
  fk_injection::impl::check_eval_points(points);

  fk_injection::HorizontalBox box =
      fk_injection::bounding_box(points.x, points.y);
  if (points.horizontal_bounds) {
    const auto &[x_min, x_max, y_min, y_max] = *points.horizontal_bounds;
    if (points.size() == 0) {
      box = { x_min, x_max, y_min, y_max };
    } else {
      box = { std::min(box.x_min, x_min), std::max(box.x_max, x_max),
              std::min(box.y_min, y_min), std::max(box.y_max, y_max) };
    }
  }
  const fk_injection::PlaneWaveResponse response(model, wave, window, box,
                                                 convention);

  const bool has_medium = !points.medium.empty();
  const bool has_traction = !points.normal_x.empty();
  std::vector<fk_injection::FieldPoint> field_points(points.size());
  std::vector<fk_injection::SeriesKey> keys(points.size());
  for (int i = 0; i < points.size(); ++i) {
    fk_injection::FieldPoint &point = field_points[i];
    point.x = points.x[i];
    point.y = points.y[i];
    point.depth = points.z[i];
    point.medium =
        has_medium
            ? points.medium[i]
            : model.locate(static_cast<double>(points.z[i]), std::nullopt, 0.0)
                  .medium;
    if (has_traction) {
      point.normal_x = points.normal_x[i];
      point.normal_y = points.normal_y[i];
      point.normal_z = points.normal_z[i];
      point.xi1 = points.lame_ratio_xi1[i];
      point.xim = points.lame_ratio_xim[i];
      point.bulk = points.lame_ratio_bulk[i];
    }
    keys[i] = { point.medium, point.depth };
  }

  return fk_injection::FieldSampler(
      response,
      std::make_shared<const fk_injection::SeriesTable>(
          fk_injection::SeriesTable::synthesize(response, keys)),
      field_points);
}
