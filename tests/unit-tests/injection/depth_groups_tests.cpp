#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/impl/depth_groups.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------
namespace depth_groups_tests_impl {

/**
 * @brief Build an EvalPoints with given x, y, z arrays and optional per-point
 *        medium selection.
 */
specfem::injection::fk::EvalPoints make_eval_points(
    const std::vector<double> &xs, const std::vector<double> &ys,
    const std::vector<double> &zs,
    const std::vector<specfem::injection::fk::point_medium> &medium = {}) {
  const int n = static_cast<int>(xs.size());

  specfem::injection::fk::EvalPoints ep;
  ep.x = Kokkos::View<type_real *, Kokkos::HostSpace>("x", n);
  ep.y = Kokkos::View<type_real *, Kokkos::HostSpace>("y", n);
  ep.z = Kokkos::View<type_real *, Kokkos::HostSpace>("z", n);
  for (int i = 0; i < n; ++i) {
    ep.x(i) = static_cast<type_real>(xs[i]);
    ep.y(i) = static_cast<type_real>(ys[i]);
    ep.z(i) = static_cast<type_real>(zs[i]);
  }

  if (!medium.empty()) {
    ep.medium =
        Kokkos::View<specfem::injection::fk::point_medium *, Kokkos::HostSpace>(
            "medium", n);
    for (int i = 0; i < n; ++i) {
      ep.medium(i) = medium[i];
    }
  }

  return ep;
}

/** @brief Elastic-only 3-layer model: two finite layers + half-space. */
specfem::injection::fk::LayeredModel make_elastic_model() {
  specfem::injection::fk::ElasticIsotropicLayer layer0;
  layer0.density = 2800;
  layer0.p_velocity = 6000;
  layer0.s_velocity = 3500;
  layer0.thickness = 10000;

  specfem::injection::fk::ElasticIsotropicLayer layer1;
  layer1.density = 2900;
  layer1.p_velocity = 6800;
  layer1.s_velocity = 3900;
  layer1.thickness = 20000;

  specfem::injection::fk::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300;
  halfspace.p_velocity = 8100;
  halfspace.s_velocity = 4500;
  halfspace.thickness = 0;

  return specfem::injection::fk::LayeredModel({},
                                              { layer0, layer1, halfspace });
}

/**
 * @brief Acoustic + elastic model: two fluid layers on top, crust + half-space.
 *
 * Layer structure (depths positive-downward):
 *   0–2000  m  : fluid vp=2000
 *   2000–3000 m: fluid vp=1000
 *   3000–13000m: elastic crust
 *   13000+     : elastic half-space
 */
specfem::injection::fk::LayeredModel make_a1_model() {
  specfem::injection::fk::AcousticLayer fluid0;
  fluid0.density = 1000;
  fluid0.p_velocity = 2000;
  fluid0.thickness = 2000;

  specfem::injection::fk::AcousticLayer fluid1;
  fluid1.density = 900;
  fluid1.p_velocity = 1000;
  fluid1.thickness = 1000;

  specfem::injection::fk::ElasticIsotropicLayer crust;
  crust.density = 2800;
  crust.p_velocity = 6000;
  crust.s_velocity = 3500;
  crust.thickness = 10000;

  specfem::injection::fk::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300;
  halfspace.p_velocity = 8100;
  halfspace.s_velocity = 4500;
  halfspace.thickness = 0;

  return specfem::injection::fk::LayeredModel({ fluid0, fluid1 },
                                              { crust, halfspace });
}

/** @brief Default incident P wave, azimuth phi=0.3 rad, theta=20 deg. */
specfem::injection::fk::IncidentWave make_wave_p() {
  specfem::injection::fk::IncidentWave wave;
  wave.type = specfem::injection::fk::incident_wave_type::p;
  wave.azimuth_phi = static_cast<type_real>(0.3);
  wave.take_off_theta = static_cast<type_real>(20.0 * 3.14159265358979 / 180.0);
  wave.origin_x = 0;
  wave.origin_y = 0;
  wave.origin_z = 0;
  wave.origin_time = 0;
  wave.amplitude = 1;
  wave.gaussian_half_duration = 0;
  return wave;
}

} // namespace depth_groups_tests_impl

// ===========================================================================
// Test 1: Clearly-separated depths → each point in its own group
// ===========================================================================
TEST(DepthGroups, DistinctDepthsFormDistinctGroups) {
  const auto model = depth_groups_tests_impl::make_elastic_model();
  const auto wave = depth_groups_tests_impl::make_wave_p();

  // Four depths spanning different layers: surface, layer0, layer1, halfspace.
  const std::vector<double> zs = { 0.0, 5000.0, 15000.0, 35000.0 };
  const int n = static_cast<int>(zs.size());
  const std::vector<double> xs(n, 0.0);
  const std::vector<double> ys(n, 0.0);

  const auto ep = depth_groups_tests_impl::make_eval_points(xs, ys, zs);

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
  const double eps = specfem::injection::fk_impl::depth_tolerance(35000.0);

  const specfem::injection::fk_impl::DepthGroups dg =
      specfem::injection::fk_impl::group_by_depth(ep, model, wave, p, eps);

  EXPECT_EQ(static_cast<int>(dg.groups.size()), n);
  // All point_group values must be distinct.
  const std::set<int> unique_groups(dg.point_group.begin(),
                                    dg.point_group.end());
  EXPECT_EQ(static_cast<int>(unique_groups.size()), n);
}

// ===========================================================================
// Test 2: Same depth, different (x,y) → single group; delays match formula
// ===========================================================================
TEST(DepthGroups, LateralPositionsShareOneGroup) {
  const auto model = depth_groups_tests_impl::make_elastic_model();
  const auto wave = depth_groups_tests_impl::make_wave_p();

  const double z_depth = 5000.0;
  const std::vector<double> xs_list = { 0.0, 1000.0, -500.0, 2500.0 };
  const std::vector<double> ys_list = { 0.0, 0.0, 1500.0, -800.0 };
  const int m = static_cast<int>(xs_list.size());
  const std::vector<double> zs(m, z_depth);

  const auto ep =
      depth_groups_tests_impl::make_eval_points(xs_list, ys_list, zs);

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
  const double eps = specfem::injection::fk_impl::depth_tolerance(z_depth);

  const specfem::injection::fk_impl::DepthGroups dg =
      specfem::injection::fk_impl::group_by_depth(ep, model, wave, p, eps);

  EXPECT_EQ(static_cast<int>(dg.groups.size()), 1);
  for (int i = 0; i < m; ++i) {
    EXPECT_EQ(dg.point_group[i], 0);
  }

  // tau = p*(x*cos(phi) + y*sin(phi)) because x0=y0=z0=0 and eta_p*(0-0)=0.
  const double phi = static_cast<double>(wave.azimuth_phi);
  for (int i = 0; i < m; ++i) {
    const double expected_tau =
        p * (xs_list[i] * std::cos(phi) + ys_list[i] * std::sin(phi));
    EXPECT_NEAR(dg.time_delay[i], expected_tau, 1e-9);
  }
}

// ===========================================================================
// Test 3: Identical depths merge; physically distinct depths separate
// ===========================================================================
TEST(DepthGroups, IdenticalDepthsMergeDistinctSeparate) {
  const auto model = depth_groups_tests_impl::make_elastic_model();
  const auto wave = depth_groups_tests_impl::make_wave_p();

  const double z_base = 5000.0;
  const double eps = specfem::injection::fk_impl::depth_tolerance(z_base);
  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);

  // Two points at the exact same depth → one group.
  const std::vector<double> zs_same = { z_base, z_base };
  // Two points half a metre apart → two groups (well beyond float round-off).
  const std::vector<double> zs_distinct = { z_base, z_base + 0.5 };
  const int n2 = 2;
  const std::vector<double> zeros(n2, 0.0);

  {
    const auto ep =
        depth_groups_tests_impl::make_eval_points(zeros, zeros, zs_same);
    const auto dg =
        specfem::injection::fk_impl::group_by_depth(ep, model, wave, p, eps);
    EXPECT_EQ(static_cast<int>(dg.groups.size()), 1)
        << "identical depths must share one group";
  }
  {
    const auto ep =
        depth_groups_tests_impl::make_eval_points(zeros, zeros, zs_distinct);
    const auto dg =
        specfem::injection::fk_impl::group_by_depth(ep, model, wave, p, eps);
    EXPECT_EQ(static_cast<int>(dg.groups.size()), 2)
        << "half-metre separation must form distinct groups";
  }
}

// ===========================================================================
// Test 4: Distinct depth levels never merge; identical depths collapse
// ===========================================================================
TEST(DepthGroups, DistinctDepthLevelsDoNotMerge) {
  const auto model = depth_groups_tests_impl::make_elastic_model();
  const auto wave = depth_groups_tests_impl::make_wave_p();

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);

  // 6 depths spaced 1000 m apart: exact grouping cannot merge them.
  {
    const int n_pts = 6;
    std::vector<double> zs(n_pts);
    for (int k = 0; k < n_pts; ++k) {
      zs[k] = 5000.0 + k * 1000.0;
    }
    const double eps = specfem::injection::fk_impl::depth_tolerance(zs.back());
    const std::vector<double> zeros(n_pts, 0.0);

    const auto ep = depth_groups_tests_impl::make_eval_points(zeros, zeros, zs);
    const auto dg =
        specfem::injection::fk_impl::group_by_depth(ep, model, wave, p, eps);
    EXPECT_EQ(static_cast<int>(dg.groups.size()), n_pts)
        << "six depths 1000 m apart must form six groups";
  }

  // 6 points at two depth levels (3 each): should collapse to 2 groups.
  {
    const double za = 5000.0;
    const double zb = 8000.0;
    const std::vector<double> zs = { za, za, za, zb, zb, zb };
    const int n_pts = static_cast<int>(zs.size());
    const double eps = specfem::injection::fk_impl::depth_tolerance(zb);
    const std::vector<double> zeros(n_pts, 0.0);

    const auto ep = depth_groups_tests_impl::make_eval_points(zeros, zeros, zs);
    const auto dg =
        specfem::injection::fk_impl::group_by_depth(ep, model, wave, p, eps);
    EXPECT_EQ(static_cast<int>(dg.groups.size()), 2)
        << "two repeated depths must collapse to two groups";
  }
}

// ===========================================================================
// Test 5: Same depth, different medium override → two groups
// ===========================================================================
TEST(DepthGroups, MediumSplitsGroups) {
  // Use the A1 model where z=3000 falls in the fluid column (0-3000 m).
  const auto model = depth_groups_tests_impl::make_a1_model();
  const auto wave = depth_groups_tests_impl::make_wave_p();

  // Use the fluid-solid interface depth (3000 m = 2000+1000) so that both
  // acoustic and elastic overrides are accepted by locate().
  const double z_test = 3000.0;
  const std::vector<double> zs = { z_test, z_test };
  const std::vector<double> zeros(2, 0.0);
  // Point 0: forced acoustic; point 1: forced elastic.
  const std::vector<specfem::injection::fk::point_medium> overrides = {
    specfem::injection::fk::point_medium::acoustic,
    specfem::injection::fk::point_medium::elastic
  };

  const auto ep =
      depth_groups_tests_impl::make_eval_points(zeros, zeros, zs, overrides);

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
  const double eps = specfem::injection::fk_impl::depth_tolerance(z_test);

  const auto dg =
      specfem::injection::fk_impl::group_by_depth(ep, model, wave, p, eps);

  EXPECT_EQ(static_cast<int>(dg.groups.size()), 2)
      << "acoustic and elastic overrides at the same depth must form two "
         "groups";

  // Find which group is acoustic and which is elastic.
  const int g0 = dg.point_group[0];
  const int g1 = dg.point_group[1];
  EXPECT_NE(g0, g1);
  EXPECT_EQ(dg.groups[g0].medium,
            specfem::injection::fk::point_medium::acoustic);
  EXPECT_EQ(dg.groups[g1].medium,
            specfem::injection::fk::point_medium::elastic);
}

// ===========================================================================
// Test 6: ray_parameter and depth_tolerance formulas
// ===========================================================================
TEST(DepthGroups, RayParameterAndTolerance) {
  const auto model = depth_groups_tests_impl::make_elastic_model();

  // Half-space velocities.
  const double vp_hs =
      static_cast<double>(model.elastic_layers().back().p_velocity);
  const double vs_hs =
      static_cast<double>(model.elastic_layers().back().s_velocity);

  const double theta_deg = 25.0;
  const double theta = theta_deg * 3.14159265358979 / 180.0;

  {
    specfem::injection::fk::IncidentWave wave;
    wave.type = specfem::injection::fk::incident_wave_type::p;
    wave.take_off_theta = static_cast<type_real>(theta);
    const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
    EXPECT_NEAR(p, std::sin(theta) / vp_hs, 1e-12);
  }
  {
    specfem::injection::fk::IncidentWave wave;
    wave.type = specfem::injection::fk::incident_wave_type::sv;
    wave.take_off_theta = static_cast<type_real>(theta);
    const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
    EXPECT_NEAR(p, std::sin(theta) / vs_hs, 1e-12);
  }

  // Depth tolerance: single-precision round-off slack only.
  const double max_depth = 20000.0;
  const float z_f = static_cast<float>(max_depth);
  const float next_f =
      std::nextafterf(z_f, std::numeric_limits<float>::infinity());
  const double expected_eps = 8.0 * static_cast<double>(next_f - z_f);

  EXPECT_NEAR(specfem::injection::fk_impl::depth_tolerance(max_depth),
              expected_eps, 1e-12);
}
