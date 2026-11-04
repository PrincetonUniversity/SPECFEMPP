#include "impl/depth_groups.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

double fk_injection::impl::depth_tolerance(double max_depth) {
  // Float round-off slack only: group points the mesh placed at the same depth
  // (identical to single precision), never points at distinct depth levels. No
  // physical distance (velocity, time step) is involved.
  const float z_f = static_cast<float>(max_depth);
  const float next_f =
      std::nextafterf(z_f, std::numeric_limits<float>::infinity());
  return 8.0 * static_cast<double>(next_f - z_f);
}

fk_injection::impl::DepthGroups
fk_injection::impl::group_by_depth(const fk_injection::EvalPoints &points,
                                   const fk_injection::LayeredModel &model,
                                   const fk_injection::IncidentWave &wave,
                                   double p, double tolerance) {
  const int n = points.size();

  // Coordinates and medium Views are host-resident; read them directly.
  const bool has_medium = (!points.medium.empty());

  // --- Per-point requested medium ---
  std::vector<std::optional<specfem::plugin::Medium>> requested(n);
  if (has_medium) {
    for (int i = 0; i < n; ++i) {
      requested[i] = points.medium[i];
    }
  }

  // --- Per-point time delay tau ---
  // tau = p*[(x-x0)*cos(phi) + (y-y0)*sin(phi)] + eta_p*(0 - z0)
  // eta_p uses the half-space P velocity regardless of wave type.
  const double phi = static_cast<double>(wave.azimuth_phi);
  const double x0 = static_cast<double>(wave.origin_x);
  const double y0 = static_cast<double>(wave.origin_y);
  const double z0 = static_cast<double>(wave.origin_z);
  const double vp_hs =
      static_cast<double>(model.elastic_layers().back().p_velocity);
  const double eta_p = std::sqrt(std::max(0.0, 1.0 / (vp_hs * vp_hs) - p * p));

  std::vector<double> time_delay(n, 0.0);
  double time_delay_min = 0.0;
  double time_delay_max = 0.0;

  if (n > 0) {
    for (int i = 0; i < n; ++i) {
      const double xi = static_cast<double>(points.x[i]);
      const double yi = static_cast<double>(points.y[i]);
      time_delay[i] =
          p * ((xi - x0) * std::cos(phi) + (yi - y0) * std::sin(phi)) +
          eta_p * (0.0 - z0);
    }
    time_delay_min = *std::min_element(time_delay.begin(), time_delay.end());
    time_delay_max = *std::max_element(time_delay.begin(), time_delay.end());
  }

  // --- Resolve each point's medium ---
  std::vector<specfem::plugin::Medium> resolved(n);
  for (int i = 0; i < n; ++i) {
    const double zi = static_cast<double>(points.z[i]);
    resolved[i] = model.locate(zi, requested[i], tolerance).medium;
  }

  // --- Sort indices by (resolved_medium, z) ascending ---
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    // Elastic groups first, independent of the SDK's enum values.
    if (resolved[a] != resolved[b]) {
      return resolved[b] == specfem::plugin::Medium::acoustic;
    }
    return static_cast<double>(points.z[a]) < static_cast<double>(points.z[b]);
  });

  // --- Anchor scan: build groups ---
  DepthGroups result;
  result.point_group.resize(n, 0);
  result.time_delay = time_delay;
  result.time_delay_min = time_delay_min;
  result.time_delay_max = time_delay_max;

  if (n == 0) {
    return result;
  }

  int current_group_id = -1;
  double z_anchor = 0.0;
  specfem::plugin::Medium group_medium = specfem::plugin::Medium::elastic;

  for (int k = 0; k < n; ++k) {
    const int idx = order[k];
    const double zi = static_cast<double>(points.z[idx]);
    const specfem::plugin::Medium med = resolved[idx];

    const bool new_medium = (k == 0) || (med != group_medium);
    const bool beyond_anchor = (k > 0) && (zi - z_anchor > tolerance);

    if (new_medium || beyond_anchor) {
      // Start a new group anchored at this point's depth.
      ++current_group_id;
      z_anchor = zi;
      group_medium = med;

      const fk_injection::LayerLocation loc =
          model.locate(z_anchor, med, tolerance);

      DepthGroup dg;
      dg.medium = loc.medium;
      dg.layer = loc.layer;
      dg.height = loc.height;
      dg.depth_below_halfspace_top = loc.depth_below_halfspace_top;
      dg.in_halfspace = loc.in_halfspace;
      dg.free_surface = loc.at_free_surface;
      dg.depth = z_anchor;
      result.groups.push_back(dg);
    }

    result.point_group[idx] = current_group_id;
  }

  return result;
}
