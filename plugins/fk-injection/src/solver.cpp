#include "fk_injection/solver.hpp"
#include "impl/depth_groups.hpp"
#include "impl/depth_series.hpp"
#include "impl/driver.hpp"
#include "plane_wave_response.hpp"
#include <algorithm>
#include <cmath>

fk_injection::FkResult
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
  const fk_injection::impl::FkSetup &setup = response.setup();

  // Largest point depth magnitude, for the float round-off grouping tolerance.
  double max_depth = 0.0;
  const int number_of_points = points.size();
  for (int i = 0; i < number_of_points; ++i) {
    max_depth =
        std::max(max_depth, std::fabs(static_cast<double>(points.z[i])));
  }

  const double tolerance = fk_injection::impl::depth_tolerance(max_depth);
  const fk_injection::impl::DepthGroups groups =
      fk_injection::impl::group_by_depth(points, model, wave, setup.p,
                                         tolerance);
  return fk_injection::impl::make_result(
      fk_injection::impl::depth_series(groups, response.states(), model, setup),
      groups, points, setup);
}
