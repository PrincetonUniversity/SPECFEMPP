#include "impl/driver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

void fk_injection::impl::check_eval_points(
    const fk_injection::EvalPoints &points) {
  const std::size_t n = points.x.size();
  auto fail = [n](const char *name, std::size_t size) {
    std::ostringstream message;
    message << "EvalPoints: " << name << " has " << size
            << " entries, expected " << n;
    throw std::runtime_error(message.str());
  };

  if (points.y.size() != n) {
    fail("y", points.y.size());
  }
  if (points.z.size() != n) {
    fail("z", points.z.size());
  }
  if (!points.medium.empty() && points.medium.size() != n) {
    fail("medium", points.medium.size());
  }

  const bool has_traction_inputs =
      !points.normal_x.empty() || !points.normal_y.empty() ||
      !points.normal_z.empty() || !points.lame_ratio_xi1.empty() ||
      !points.lame_ratio_xim.empty() || !points.lame_ratio_bulk.empty();
  if (has_traction_inputs) {
    const std::pair<const char *, std::size_t> traction_inputs[] = {
      { "normal_x", points.normal_x.size() },
      { "normal_y", points.normal_y.size() },
      { "normal_z", points.normal_z.size() },
      { "lame_ratio_xi1", points.lame_ratio_xi1.size() },
      { "lame_ratio_xim", points.lame_ratio_xim.size() },
      { "lame_ratio_bulk", points.lame_ratio_bulk.size() },
    };
    for (const auto &[name, size] : traction_inputs) {
      if (size != n) {
        fail(name, size);
      }
    }
  }
}

void fk_injection::impl::check_supported_slowness(
    const fk_injection::LayeredModel &model, double p) {
  // The negated comparisons also reject NaN.
  if (!(p > 0.0)) {
    std::ostringstream message;
    message << "FK needs a positive ray parameter, got p = " << p
            << " s/m; vertical incidence is not supported";
    throw specfem::plugin::Unsupported(message.str());
  }

  double fastest = 0.0;
  for (const auto &layer : model.fluid_layers()) {
    fastest = std::max(fastest, static_cast<double>(layer.p_velocity));
  }
  for (const auto &layer : model.elastic_layers()) {
    fastest = std::max({ fastest, static_cast<double>(layer.p_velocity),
                         static_cast<double>(layer.s_velocity) });
  }
  const double limit = 1.0 / fastest;
  if (!(p < limit)) {
    std::ostringstream message;
    message << "FK needs p below " << limit << " s/m (1/v for the fastest "
            << "velocity in the model, " << fastest << " m/s), got p = " << p
            << " s/m; critical and post-critical incidence are not supported, "
            << "which limits SV to take-off angles below asin(vs/vp)";
    throw specfem::plugin::Unsupported(message.str());
  }
}
