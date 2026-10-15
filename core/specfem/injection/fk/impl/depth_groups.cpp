#include "specfem/injection/fk/impl/depth_groups.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

double specfem::injection::fk_impl::ray_parameter(
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk::IncidentWave &wave) {
  const specfem::injection::fk::ElasticIsotropicLayer &hs =
      model.elastic_layers().back();
  const double v_hs =
      (wave.type == specfem::injection::fk::incident_wave_type::p)
          ? static_cast<double>(hs.p_velocity)
          : static_cast<double>(hs.s_velocity);
  return std::sin(static_cast<double>(wave.take_off_theta)) / v_hs;
}

double specfem::injection::fk_impl::depth_tolerance(double max_depth) {
  // Float round-off slack only: group points the mesh placed at the same depth
  // (identical to single precision), never points at distinct depth levels. No
  // physical distance (velocity, time step) is involved.
  const float z_f = static_cast<float>(max_depth);
  const float next_f =
      std::nextafterf(z_f, std::numeric_limits<float>::infinity());
  return 8.0 * static_cast<double>(next_f - z_f);
}

specfem::injection::fk_impl::DepthGroups
specfem::injection::fk_impl::group_by_depth(
    const specfem::injection::fk::EvalPoints &points,
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk::IncidentWave &wave, double p,
    double tolerance) {
  const int n = points.size();

  // Coordinates and medium Views are host-resident; read them directly.
  const bool has_medium = (points.medium.extent(0) > 0);

  // --- Per-point requested medium ---
  std::vector<specfem::injection::fk::point_medium> requested(n);
  for (int i = 0; i < n; ++i) {
    requested[i] = has_medium
                       ? points.medium(i)
                       : specfem::injection::fk::point_medium::from_depth;
  }

  // --- Per-point time delay tau ---
  // tau = p*[(x-x0)*cos(phi) + (y-y0)*sin(phi)] + eta_p*(z_ref - z0)
  // eta_p uses the half-space P velocity regardless of wave type.
  //
  // The vertical anchor z_ref is the HALF-SPACE TOP, not the model top: the
  // FK solve's time zero is where the incident wave crosses the half-space
  // top (SPECFEM3D's Z_REF_for_FK; couple_with_injection.f90 shifts both the
  // points and the origin into that frame). The two coincide only when the
  // model has no finite layers.
  const double phi = static_cast<double>(wave.azimuth_phi);
  const double x0 = static_cast<double>(wave.origin_x);
  const double y0 = static_cast<double>(wave.origin_y);
  const double z0 = static_cast<double>(wave.origin_z);
  // Half-space top in the origin's frame (z up, model top 0): depth below the
  // model top, negated.
  const double z_ref = -static_cast<double>(model.layer_top_depths().back());
  const double vp_hs =
      static_cast<double>(model.elastic_layers().back().p_velocity);
  const double eta_p = std::sqrt(std::max(0.0, 1.0 / (vp_hs * vp_hs) - p * p));

  std::vector<double> time_delay(n, 0.0);
  double time_delay_min = 0.0;
  double time_delay_max = 0.0;

  if (n > 0) {
    for (int i = 0; i < n; ++i) {
      const double xi = static_cast<double>(points.x(i));
      const double yi = static_cast<double>(points.y(i));
      time_delay[i] =
          p * ((xi - x0) * std::cos(phi) + (yi - y0) * std::sin(phi)) +
          eta_p * (z_ref - z0);
    }
    time_delay_min = *std::min_element(time_delay.begin(), time_delay.end());
    time_delay_max = *std::max_element(time_delay.begin(), time_delay.end());
  }

  // --- Resolve each point's medium ---
  std::vector<specfem::injection::fk::point_medium> resolved(n);
  for (int i = 0; i < n; ++i) {
    const double zi = static_cast<double>(points.z(i));
    resolved[i] = model.locate(zi, requested[i], tolerance).medium;
  }

  // --- Sort indices by (resolved_medium, z) ascending ---
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const int ma = static_cast<int>(resolved[a]);
    const int mb = static_cast<int>(resolved[b]);
    if (ma != mb) {
      return ma < mb;
    }
    return static_cast<double>(points.z(a)) < static_cast<double>(points.z(b));
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
  specfem::injection::fk::point_medium group_medium =
      specfem::injection::fk::point_medium::from_depth;

  for (int k = 0; k < n; ++k) {
    const int idx = order[k];
    const double zi = static_cast<double>(points.z(idx));
    const specfem::injection::fk::point_medium med = resolved[idx];

    const bool new_medium = (k == 0) || (med != group_medium);
    const bool beyond_anchor = (k > 0) && (zi - z_anchor > tolerance);

    if (new_medium || beyond_anchor) {
      // Start a new group anchored at this point's depth.
      ++current_group_id;
      z_anchor = zi;
      group_medium = med;

      const specfem::injection::fk::LayerLocation loc =
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
