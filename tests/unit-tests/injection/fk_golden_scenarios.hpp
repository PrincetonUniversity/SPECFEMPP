#pragma once

#include "specfem/injection/fk.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>
#include <vector>

namespace specfem {
namespace testing {
namespace fk_golden {

/**
 * @brief Fully-specified input + metadata for one golden-capture scenario.
 *
 * Built by one of the five builder functions below; all device Views are
 * allocated and deep-copied from host mirrors so the struct is ready to pass
 * directly to @c specfem::injection::fk::solve.
 */
struct GoldenScenario {
  std::string id; ///< Short label ("E1", "E2", ..., "A1")

  specfem::injection::fk::LayeredModel model;
  specfem::injection::fk::IncidentWave wave;
  specfem::injection::fk::TimeWindow window;
  specfem::injection::fk::EvalPoints points;

  bool compute_traction;
  specfem::injection::fk::field_derivative derivative;

  /// Host copies of point coordinates, kept for file metadata.
  std::vector<double> point_z;
  std::vector<double> point_x;
  std::vector<double> point_y;

  /**
   * @brief Resolved medium per point: 0 = elastic, 1 = acoustic.
   *
   * Acoustic when the point depth falls inside a fluid layer OR when the
   * per-point medium forces @c point_medium::acoustic.
   */
  std::vector<int> point_medium;
};

// ---------------------------------------------------------------------------
// Module-internal helpers
// ---------------------------------------------------------------------------

namespace fk_golden_impl {

// Four unit normals cycled by point index.
constexpr int kNormalCycle = 4;
constexpr double kNormals[kNormalCycle][3] = {
  { 0.0, 0.0, 1.0 },
  { 1.0, 0.0, 0.0 },
  { 0.0, 1.0, 0.0 },
  { 0.6, 0.0, 0.8 },
};

/**
 * @brief Lamé ratios for a given elastic layer.
 *
 * These are deterministic test fixtures shared by capture and the later
 * comparison run; they are not an authoritative physical derivation.
 */
struct LameRatios {
  double xi1;
  double xim;
  double bulk;
};

inline LameRatios lame_from_elastic(double rho, double vp, double vs) {
  const double mu = rho * vs * vs;
  const double lambda = rho * vp * vp - 2.0 * mu;
  return { (vp * vp) / (vs * vs), 0.5, lambda / (2.0 * (lambda + mu)) };
}

/**
 * @brief Build host-resident EvalPoints from parallel host coordinate vectors.
 *
 * @param xs, ys, zs   Coordinate vectors (equal length).
 * @param normals_nx/ny/nz  Per-point normal vectors (same length; empty = no
 *                          normals allocated).
 * @param xi1s, xims, bulks Per-point Lamé ratios (same length; empty = no
 *                           Lamé Views allocated).
 * @param medium  Per-point medium selection (empty = all from_depth).
 */
inline specfem::injection::fk::EvalPoints build_eval_points(
    const std::vector<double> &xs, const std::vector<double> &ys,
    const std::vector<double> &zs, const std::vector<double> &normals_nx,
    const std::vector<double> &normals_ny,
    const std::vector<double> &normals_nz, const std::vector<double> &xi1s,
    const std::vector<double> &xims, const std::vector<double> &bulks,
    const std::vector<specfem::injection::fk::point_medium> &medium) {
  const int n = static_cast<int>(xs.size());

  specfem::injection::fk::EvalPoints ep;
  ep.x = Kokkos::View<type_real *, Kokkos::HostSpace>("ep_x", n);
  ep.y = Kokkos::View<type_real *, Kokkos::HostSpace>("ep_y", n);
  ep.z = Kokkos::View<type_real *, Kokkos::HostSpace>("ep_z", n);
  for (int i = 0; i < n; ++i) {
    ep.x(i) = static_cast<type_real>(xs[i]);
    ep.y(i) = static_cast<type_real>(ys[i]);
    ep.z(i) = static_cast<type_real>(zs[i]);
  }

  if (!normals_nx.empty()) {
    ep.normal_x = Kokkos::View<type_real *, Kokkos::HostSpace>("ep_nx", n);
    ep.normal_y = Kokkos::View<type_real *, Kokkos::HostSpace>("ep_ny", n);
    ep.normal_z = Kokkos::View<type_real *, Kokkos::HostSpace>("ep_nz", n);
    for (int i = 0; i < n; ++i) {
      ep.normal_x(i) = static_cast<type_real>(normals_nx[i]);
      ep.normal_y(i) = static_cast<type_real>(normals_ny[i]);
      ep.normal_z(i) = static_cast<type_real>(normals_nz[i]);
    }
  }

  if (!xi1s.empty()) {
    ep.lame_ratio_xi1 =
        Kokkos::View<type_real *, Kokkos::HostSpace>("ep_xi1", n);
    ep.lame_ratio_xim =
        Kokkos::View<type_real *, Kokkos::HostSpace>("ep_xim", n);
    ep.lame_ratio_bulk =
        Kokkos::View<type_real *, Kokkos::HostSpace>("ep_bulk", n);
    for (int i = 0; i < n; ++i) {
      ep.lame_ratio_xi1(i) = static_cast<type_real>(xi1s[i]);
      ep.lame_ratio_xim(i) = static_cast<type_real>(xims[i]);
      ep.lame_ratio_bulk(i) = static_cast<type_real>(bulks[i]);
    }
  }

  if (!medium.empty()) {
    ep.medium =
        Kokkos::View<specfem::injection::fk::point_medium *, Kokkos::HostSpace>(
            "ep_medium", n);
    for (int i = 0; i < n; ++i) {
      ep.medium(i) = medium[i];
    }
  }

  return ep;
}

/**
 * @brief Determine which elastic layer a depth @p z falls into.
 *
 * Returns the 0-based index into @c elastic_layers.  The fluid block is
 * excluded; this function is only called after determining the point is in the
 * elastic block.
 */
inline int
elastic_layer_for_depth(const specfem::injection::fk::LayeredModel &model,
                        double z) {
  const auto &elayers = model.elastic_layers();
  const int nfluid = model.number_of_fluid_layers();
  const auto &tops = model.layer_top_depths();
  const int nelastic = static_cast<int>(elayers.size());

  // tops has global order: [fluid layers..., elastic layers...]
  for (int i = 0; i < nelastic - 1; ++i) {
    const double top = static_cast<double>(tops[nfluid + i]);
    const double bot = static_cast<double>(tops[nfluid + i]) +
                       static_cast<double>(elayers[i].thickness);
    if (z >= top && z < bot) {
      return i;
    }
  }
  // Falls in the half-space (last elastic layer)
  return nelastic - 1;
}

/**
 * @brief Determine whether a point at depth @p z falls inside a fluid layer.
 */
inline bool is_fluid_depth(const specfem::injection::fk::LayeredModel &model,
                           double z) {
  const auto &flayers = model.fluid_layers();
  const auto &tops = model.layer_top_depths();
  const int nfluid = static_cast<int>(flayers.size());
  for (int i = 0; i < nfluid; ++i) {
    const double top = static_cast<double>(tops[i]);
    const double bot = top + static_cast<double>(flayers[i].thickness);
    if (z >= top && z < bot) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Fill normals and Lamé-ratio vectors for @p n points.
 *
 * Normals cycle through kNormals by point index.  Lamé ratios are derived
 * from the elastic layer each point resolves to; fluid points get zeros.
 *
 * @param model     Layered model used to look up material parameters.
 * @param zs        Point depths.
 * @param overrides Per-point medium selection; may be empty (all from_depth).
 * @param nx/ny/nz  Output normal vectors (filled on return).
 * @param xi1s/xims/bulks  Output Lamé ratio vectors (filled on return).
 * @param medium    Output medium flags: 0 = elastic, 1 = acoustic.
 */
inline void fill_normals_and_lame(
    const specfem::injection::fk::LayeredModel &model,
    const std::vector<double> &zs,
    const std::vector<specfem::injection::fk::point_medium> &overrides,
    std::vector<double> &nx, std::vector<double> &ny, std::vector<double> &nz,
    std::vector<double> &xi1s, std::vector<double> &xims,
    std::vector<double> &bulks, std::vector<int> &medium) {
  const int n = static_cast<int>(zs.size());
  nx.resize(n);
  ny.resize(n);
  nz.resize(n);
  xi1s.resize(n);
  xims.resize(n);
  bulks.resize(n);
  medium.resize(n);

  for (int i = 0; i < n; ++i) {
    // Cycling normals
    const int cycle = i % kNormalCycle;
    nx[i] = kNormals[cycle][0];
    ny[i] = kNormals[cycle][1];
    nz[i] = kNormals[cycle][2];

    // Resolved medium
    const specfem::injection::fk::point_medium ov =
        (!overrides.empty()) ? overrides[i]
                             : specfem::injection::fk::point_medium::from_depth;
    bool acoustic = false;
    if (ov == specfem::injection::fk::point_medium::acoustic) {
      acoustic = true;
    } else if (ov == specfem::injection::fk::point_medium::elastic) {
      acoustic = false;
    } else {
      acoustic = is_fluid_depth(model, zs[i]);
    }
    medium[i] = acoustic ? 1 : 0;

    if (acoustic) {
      xi1s[i] = 0.0;
      xims[i] = 0.0;
      bulks[i] = 0.0;
    } else {
      const int eidx = elastic_layer_for_depth(model, zs[i]);
      const auto &el = model.elastic_layers()[eidx];
      const LameRatios lr = lame_from_elastic(
          static_cast<double>(el.density), static_cast<double>(el.p_velocity),
          static_cast<double>(el.s_velocity));
      xi1s[i] = lr.xi1;
      xims[i] = lr.xim;
      bulks[i] = lr.bulk;
    }
  }
}

} // namespace fk_golden_impl

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

namespace fk_golden_constants {
constexpr double pi = 3.14159265358979323846;
constexpr double azimuth_phi = 30.0 * pi / 180.0; ///< 30° in radians
constexpr double gaussian_half_duration = 1.0;
constexpr double origin_time = -5.05;
constexpr double amplitude = 1.0;
constexpr double origin_x = 0.0;
constexpr double origin_y = 0.0;
constexpr double origin_z = 0.0;
constexpr double frequency_sampling = 10.0;
constexpr double time_window_length = 40.0;
constexpr double frequency_max = 1.0;
} // namespace fk_golden_constants

// ---------------------------------------------------------------------------
// Layered model builders
// ---------------------------------------------------------------------------

namespace fk_golden_impl {

/**
 * @brief Pure-elastic 3-layer model used by scenarios E1–E4.
 *
 * Layer order (top to bottom):
 *   L0  rho=2800  vp=6000  vs=3500  H=10000 m   (top at z=0)
 *   L1  rho=2900  vp=6800  vs=3900  H=20000 m   (top at z=10000)
 *   HS  rho=3300  vp=8100  vs=4500  H=0          (top at z=30000)
 */
inline specfem::injection::fk::LayeredModel make_elastic_model() {
  specfem::injection::fk::ElasticIsotropicLayer l0, l1, hs;
  l0.density = 2800;
  l0.p_velocity = 6000;
  l0.s_velocity = 3500;
  l0.thickness = 10000;
  l1.density = 2900;
  l1.p_velocity = 6800;
  l1.s_velocity = 3900;
  l1.thickness = 20000;
  hs.density = 3300;
  hs.p_velocity = 8100;
  hs.s_velocity = 4500;
  hs.thickness = 0;
  return specfem::injection::fk::LayeredModel({}, { l0, l1, hs });
}

/**
 * @brief Ocean-over-crust model used by scenario A1.
 *
 * Layer order (top to bottom):
 *   F0  rho=1020  vp=1500  H=2000 m             (top at z=0)
 *   F1  rho=1020  vp=1520  H=1000 m             (top at z=2000)
 *   C   rho=2800  vp=6000  vs=3500  H=10000 m   (top at z=3000)
 *   HS  rho=3300  vp=8100  vs=4500  H=0          (top at z=13000)
 */
inline specfem::injection::fk::LayeredModel make_acoustic_elastic_model() {
  specfem::injection::fk::AcousticLayer f0, f1;
  f0.density = 1020;
  f0.p_velocity = 1500;
  f0.thickness = 2000;
  f1.density = 1020;
  f1.p_velocity = 1520;
  f1.thickness = 1000;

  specfem::injection::fk::ElasticIsotropicLayer crust, hs;
  crust.density = 2800;
  crust.p_velocity = 6000;
  crust.s_velocity = 3500;
  crust.thickness = 10000;
  hs.density = 3300;
  hs.p_velocity = 8100;
  hs.s_velocity = 4500;
  hs.thickness = 0;

  return specfem::injection::fk::LayeredModel({ f0, f1 }, { crust, hs });
}

} // namespace fk_golden_impl

// ---------------------------------------------------------------------------
// Scenario builders
// ---------------------------------------------------------------------------

/**
 * @brief E1 — elastic model, incident P, theta=20°, dt=0.1, 300 steps,
 *        traction, displacement.
 *
 * Points span the model vertically plus one lateral offset point whose
 * horizontal distance is chosen so the FK time delay equals 1.3 / p_hs,
 * giving a lateral delay of ≈ 13 coarse steps.
 */
inline GoldenScenario scenario_e1() {
  using namespace fk_golden_constants;
  using namespace fk_golden_impl;

  GoldenScenario sc;
  sc.id = "E1";
  sc.model = make_elastic_model();
  sc.compute_traction = true;
  sc.derivative = specfem::injection::fk::field_derivative::displacement;

  sc.wave.type = specfem::injection::fk::incident_wave_type::p;
  sc.wave.azimuth_phi = static_cast<type_real>(azimuth_phi);
  sc.wave.take_off_theta = static_cast<type_real>(20.0 * pi / 180.0);
  sc.wave.origin_x = static_cast<type_real>(origin_x);
  sc.wave.origin_y = static_cast<type_real>(origin_y);
  sc.wave.origin_z = static_cast<type_real>(origin_z);
  sc.wave.origin_time = static_cast<type_real>(origin_time);
  sc.wave.amplitude = static_cast<type_real>(amplitude);
  sc.wave.gaussian_half_duration =
      static_cast<type_real>(gaussian_half_duration);

  sc.window.dt = static_cast<type_real>(0.1);
  sc.window.nstep = 300;
  sc.window.frequency_max = static_cast<type_real>(frequency_max);
  sc.window.frequency_sampling = static_cast<type_real>(frequency_sampling);
  sc.window.time_window_length = static_cast<type_real>(time_window_length);

  // Lateral point: p = sin(theta)/vp_hs; d = 1.3/p => lateral delay ≈ 13*dt
  const double theta_e1 = 20.0 * pi / 180.0;
  const double vp_hs = 8100.0;
  const double p_ray = std::sin(theta_e1) / vp_hs;
  const double d_lat = 1.3 / p_ray;

  sc.point_z = { 0, 5000, 10000, 10000.175, 30000, 35000, 5000, 5000, 5000 };
  sc.point_x = {
    0, 0, 0, 0, 0, 0, 20000, -20000, d_lat * std::cos(azimuth_phi)
  };
  sc.point_y = { 0, 0, 0, 0, 0, 0, 0, 10000, d_lat * std::sin(azimuth_phi) };

  const int n = static_cast<int>(sc.point_z.size());
  std::vector<specfem::injection::fk::point_medium> overrides; // all auto — no
                                                               // overrides
                                                               // needed
  std::vector<double> nx, ny, nz, xi1s, xims, bulks;
  fill_normals_and_lame(sc.model, sc.point_z, overrides, nx, ny, nz, xi1s, xims,
                        bulks, sc.point_medium);

  sc.points = build_eval_points(sc.point_x, sc.point_y, sc.point_z, nx, ny, nz,
                                xi1s, xims, bulks, overrides);
  return sc;
}

/**
 * @brief E2 — elastic model, incident SV, theta=20°, dt=0.05, 600 steps,
 *        traction, displacement.
 */
inline GoldenScenario scenario_e2() {
  using namespace fk_golden_constants;
  using namespace fk_golden_impl;

  GoldenScenario sc;
  sc.id = "E2";
  sc.model = make_elastic_model();
  sc.compute_traction = true;
  sc.derivative = specfem::injection::fk::field_derivative::displacement;

  sc.wave.type = specfem::injection::fk::incident_wave_type::sv;
  sc.wave.azimuth_phi = static_cast<type_real>(azimuth_phi);
  sc.wave.take_off_theta = static_cast<type_real>(20.0 * pi / 180.0);
  sc.wave.origin_x = static_cast<type_real>(origin_x);
  sc.wave.origin_y = static_cast<type_real>(origin_y);
  sc.wave.origin_z = static_cast<type_real>(origin_z);
  sc.wave.origin_time = static_cast<type_real>(origin_time);
  sc.wave.amplitude = static_cast<type_real>(amplitude);
  sc.wave.gaussian_half_duration =
      static_cast<type_real>(gaussian_half_duration);

  sc.window.dt = static_cast<type_real>(0.05);
  sc.window.nstep = 600;
  sc.window.frequency_max = static_cast<type_real>(frequency_max);
  sc.window.frequency_sampling = static_cast<type_real>(frequency_sampling);
  sc.window.time_window_length = static_cast<type_real>(time_window_length);

  sc.point_z = { 0, 5000, 10000, 35000, 5000 };
  sc.point_x = { 0, 0, 0, 0, 20000 };
  sc.point_y = { 0, 0, 0, 0, 0 };

  std::vector<specfem::injection::fk::point_medium> overrides;
  std::vector<double> nx, ny, nz, xi1s, xims, bulks;
  fill_normals_and_lame(sc.model, sc.point_z, overrides, nx, ny, nz, xi1s, xims,
                        bulks, sc.point_medium);

  sc.points = build_eval_points(sc.point_x, sc.point_y, sc.point_z, nx, ny, nz,
                                xi1s, xims, bulks, overrides);
  return sc;
}

/**
 * @brief E3 — elastic model, incident P, theta=20°, dt=0.05, 600 steps,
 *        traction, displacement.  tau=0 subset of E1 (no lateral points).
 */
inline GoldenScenario scenario_e3() {
  using namespace fk_golden_constants;
  using namespace fk_golden_impl;

  GoldenScenario sc;
  sc.id = "E3";
  sc.model = make_elastic_model();
  sc.compute_traction = true;
  sc.derivative = specfem::injection::fk::field_derivative::displacement;

  sc.wave.type = specfem::injection::fk::incident_wave_type::p;
  sc.wave.azimuth_phi = static_cast<type_real>(azimuth_phi);
  sc.wave.take_off_theta = static_cast<type_real>(20.0 * pi / 180.0);
  sc.wave.origin_x = static_cast<type_real>(origin_x);
  sc.wave.origin_y = static_cast<type_real>(origin_y);
  sc.wave.origin_z = static_cast<type_real>(origin_z);
  sc.wave.origin_time = static_cast<type_real>(origin_time);
  sc.wave.amplitude = static_cast<type_real>(amplitude);
  sc.wave.gaussian_half_duration =
      static_cast<type_real>(gaussian_half_duration);

  sc.window.dt = static_cast<type_real>(0.05);
  sc.window.nstep = 600;
  sc.window.frequency_max = static_cast<type_real>(frequency_max);
  sc.window.frequency_sampling = static_cast<type_real>(frequency_sampling);
  sc.window.time_window_length = static_cast<type_real>(time_window_length);

  sc.point_z = { 0, 5000, 10000, 30000, 35000 };
  sc.point_x = { 0, 0, 0, 0, 0 };
  sc.point_y = { 0, 0, 0, 0, 0 };

  std::vector<specfem::injection::fk::point_medium> overrides;
  std::vector<double> nx, ny, nz, xi1s, xims, bulks;
  fill_normals_and_lame(sc.model, sc.point_z, overrides, nx, ny, nz, xi1s, xims,
                        bulks, sc.point_medium);

  sc.points = build_eval_points(sc.point_x, sc.point_y, sc.point_z, nx, ny, nz,
                                xi1s, xims, bulks, overrides);
  return sc;
}

/**
 * @brief E4 — elastic model, incident P, theta=20°, dt=0.1, 300 steps,
 *        traction, VELOCITY derivative.
 */
inline GoldenScenario scenario_e4() {
  using namespace fk_golden_constants;
  using namespace fk_golden_impl;

  GoldenScenario sc;
  sc.id = "E4";
  sc.model = make_elastic_model();
  sc.compute_traction = true;
  sc.derivative = specfem::injection::fk::field_derivative::velocity;

  sc.wave.type = specfem::injection::fk::incident_wave_type::p;
  sc.wave.azimuth_phi = static_cast<type_real>(azimuth_phi);
  sc.wave.take_off_theta = static_cast<type_real>(20.0 * pi / 180.0);
  sc.wave.origin_x = static_cast<type_real>(origin_x);
  sc.wave.origin_y = static_cast<type_real>(origin_y);
  sc.wave.origin_z = static_cast<type_real>(origin_z);
  sc.wave.origin_time = static_cast<type_real>(origin_time);
  sc.wave.amplitude = static_cast<type_real>(amplitude);
  sc.wave.gaussian_half_duration =
      static_cast<type_real>(gaussian_half_duration);

  sc.window.dt = static_cast<type_real>(0.1);
  sc.window.nstep = 300;
  sc.window.frequency_max = static_cast<type_real>(frequency_max);
  sc.window.frequency_sampling = static_cast<type_real>(frequency_sampling);
  sc.window.time_window_length = static_cast<type_real>(time_window_length);

  sc.point_z = { 0, 5000, 30000 };
  sc.point_x = { 0, 0, 0 };
  sc.point_y = { 0, 0, 0 };

  std::vector<specfem::injection::fk::point_medium> overrides;
  std::vector<double> nx, ny, nz, xi1s, xims, bulks;
  fill_normals_and_lame(sc.model, sc.point_z, overrides, nx, ny, nz, xi1s, xims,
                        bulks, sc.point_medium);

  sc.points = build_eval_points(sc.point_x, sc.point_y, sc.point_z, nx, ny, nz,
                                xi1s, xims, bulks, overrides);
  return sc;
}

/**
 * @brief A1 — acoustic-over-elastic model, incident P, theta=15°, dt=0.1,
 *        300 steps, traction, displacement.
 *
 * The three points at z=3000 (fluid/solid interface) force acoustic, elastic,
 * and from_depth respectively to exercise the medium dispatch.
 */
inline GoldenScenario scenario_a1() {
  using namespace fk_golden_constants;
  using namespace fk_golden_impl;

  GoldenScenario sc;
  sc.id = "A1";
  sc.model = make_acoustic_elastic_model();
  sc.compute_traction = true;
  sc.derivative = specfem::injection::fk::field_derivative::displacement;

  sc.wave.type = specfem::injection::fk::incident_wave_type::p;
  sc.wave.azimuth_phi = static_cast<type_real>(azimuth_phi);
  sc.wave.take_off_theta = static_cast<type_real>(15.0 * pi / 180.0);
  sc.wave.origin_x = static_cast<type_real>(origin_x);
  sc.wave.origin_y = static_cast<type_real>(origin_y);
  sc.wave.origin_z = static_cast<type_real>(origin_z);
  sc.wave.origin_time = static_cast<type_real>(origin_time);
  sc.wave.amplitude = static_cast<type_real>(amplitude);
  sc.wave.gaussian_half_duration =
      static_cast<type_real>(gaussian_half_duration);

  sc.window.dt = static_cast<type_real>(0.1);
  sc.window.nstep = 300;
  sc.window.frequency_max = static_cast<type_real>(frequency_max);
  sc.window.frequency_sampling = static_cast<type_real>(frequency_sampling);
  sc.window.time_window_length = static_cast<type_real>(time_window_length);

  // Three z=3000 points at the fluid/solid interface with different overrides.
  // Index layout: 0-1=fluid, 2=fluid/iface, 3=fluid/iface/solid,
  // 4=fluid/iface/auto,
  //               5-6-7=elastic, 8=lateral
  sc.point_z = { 0, 1000, 2000, 3000, 3000, 3000, 8000, 13000, 15000, 1000 };
  sc.point_x = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 15000 };
  sc.point_y = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  const int n = static_cast<int>(sc.point_z.size());
  // Overrides: only the three z=3000 points (indices 3,4,5) force a medium.
  std::vector<specfem::injection::fk::point_medium> overrides(
      n, specfem::injection::fk::point_medium::from_depth);
  overrides[3] =
      specfem::injection::fk::point_medium::acoustic; // force acoustic
  overrides[4] = specfem::injection::fk::point_medium::elastic; // force elastic
  overrides[5] =
      specfem::injection::fk::point_medium::from_depth; // use layer table

  std::vector<double> nx, ny, nz, xi1s, xims, bulks;
  fill_normals_and_lame(sc.model, sc.point_z, overrides, nx, ny, nz, xi1s, xims,
                        bulks, sc.point_medium);

  sc.points = build_eval_points(sc.point_x, sc.point_y, sc.point_z, nx, ny, nz,
                                xi1s, xims, bulks, overrides);
  return sc;
}

/**
 * @brief Build all five golden scenarios in canonical order.
 * @return {E1, E2, E3, E4, A1}
 */
inline std::vector<GoldenScenario> all_scenarios() {
  return { scenario_e1(), scenario_e2(), scenario_e3(), scenario_e4(),
           scenario_a1() };
}

} // namespace fk_golden
} // namespace testing
} // namespace specfem
