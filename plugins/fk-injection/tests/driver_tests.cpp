#include "fk_injection/fk_injection.hpp"
#include "impl/driver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using fk_injection::type_real;

// ---------------------------------------------------------------------------
// Helper: build a host-side EvalPoints with n points at given z-coords.
// Normal vectors point in +z direction; Lamé ratios set to 1.
// ---------------------------------------------------------------------------
static fk_injection::EvalPoints
make_eval_points(const std::vector<double> &z_coords, bool with_normals = false,
                 bool with_lame = false) {
  const int n = static_cast<int>(z_coords.size());

  fk_injection::EvalPoints ep;
  ep.x.resize(n);
  ep.y.resize(n);
  ep.z.resize(n);
  for (int i = 0; i < n; ++i) {
    ep.x[i] = 0;
    ep.y[i] = 0;
    ep.z[i] = static_cast<type_real>(z_coords[i]);
  }

  if (with_normals) {
    ep.normal_x.resize(n);
    ep.normal_y.resize(n);
    ep.normal_z.resize(n);
    for (int i = 0; i < n; ++i) {
      ep.normal_x[i] = 0;
      ep.normal_y[i] = 0;
      ep.normal_z[i] = 1;
    }
  }

  if (with_lame) {
    ep.lame_ratio_xi1.resize(n);
    ep.lame_ratio_xim.resize(n);
    ep.lame_ratio_bulk.resize(n);
    for (int i = 0; i < n; ++i) {
      ep.lame_ratio_xi1[i] = 2; // xi1 = lambda/mu + 2
      ep.lame_ratio_xim[i] = 1; // xim = mu / (2 * mu)
      ep.lame_ratio_bulk[i] = static_cast<type_real>(0.333); // bulk Lame ratio
    }
  }

  return ep;
}

// ---------------------------------------------------------------------------
// Check that every entry is finite and that at least one entry has
// non-negligible absolute value.
// ---------------------------------------------------------------------------
static bool all_finite_not_all_zero(const std::vector<type_real> &values) {
  bool all_finite = true;
  double max_abs = 0.0;
  for (const type_real value : values) {
    const double v = static_cast<double>(value);
    if (!std::isfinite(v)) {
      all_finite = false;
    }
    if (std::fabs(v) > max_abs)
      max_abs = std::fabs(v);
  }
  return all_finite && (max_abs > 1.0e-30);
}

// ===========================================================================
// Test 1: ElasticHalfSpaceRunsFinite
// 2-layer elastic model (crust over mantle half-space), incident P.
// ===========================================================================
TEST(FkDriver, ElasticHalfSpaceRunsFinite) {
  // Crust layer: vp=6000, vs=3500, rho=2800, H=35000 m
  fk_injection::ElasticIsotropicLayer crust;
  crust.p_velocity = 6000;
  crust.s_velocity = 3500;
  crust.density = 2800;
  crust.thickness = 35000;

  // Mantle half-space: vp=8100, vs=4500, rho=3300
  fk_injection::ElasticIsotropicLayer mantle;
  mantle.p_velocity = 8100;
  mantle.s_velocity = 4500;
  mantle.density = 3300;
  mantle.thickness = 0; // half-space; thickness ignored

  fk_injection::LayeredModel model({}, { crust, mantle });

  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.azimuth_phi = 0;
  wave.take_off_theta = static_cast<type_real>(20.0 * 3.14159265 / 180.0);
  wave.origin_x = 0;
  wave.origin_y = 0;
  wave.origin_z = 0;
  wave.origin_time = 5;
  wave.amplitude = 1;
  wave.gaussian_half_duration = 2;

  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 256;
  window.frequency_max = 1;
  window.frequency_sampling = 2;
  window.time_window_length = 40;

  // Points at various depths in the crust and just below the surface.
  const auto ep = make_eval_points({ 0, 5000, 15000, 30000, 40000 });

  const auto result = fk_injection::solve(
      model, wave, window, ep, fk_injection::field_derivative::displacement);

  // Shape checks
  EXPECT_EQ(result.number_of_points(), 5);
  EXPECT_GT(result.coefficient_count(), 0);

  std::vector<type_real> out(
      static_cast<std::size_t>(result.number_of_points()) *
      fk_injection::number_of_sample_components * window.nstep);
  result.sample(0, window.nstep, out);

  EXPECT_TRUE(all_finite_not_all_zero(out))
      << "sampled output is NaN/Inf or all-zero";
}

// ===========================================================================
// Test 2: AcousticElasticRunsFinite
// Ocean (fluid) + crust + mantle, incident P.  Points in water AND solid.
// ===========================================================================
TEST(FkDriver, AcousticElasticRunsFinite) {
  // Ocean layer: vp=1500, vs=0, rho=1025, H=4000 m
  fk_injection::AcousticLayer ocean;
  ocean.p_velocity = 1500;
  ocean.density = 1025;
  ocean.thickness = 4000;

  // Crust: vp=6000, vs=3500, rho=2800, H=35000 m
  fk_injection::ElasticIsotropicLayer crust;
  crust.p_velocity = 6000;
  crust.s_velocity = 3500;
  crust.density = 2800;
  crust.thickness = 35000;

  // Mantle half-space
  fk_injection::ElasticIsotropicLayer mantle;
  mantle.p_velocity = 8100;
  mantle.s_velocity = 4500;
  mantle.density = 3300;
  mantle.thickness = 0;

  fk_injection::LayeredModel model({ ocean }, { crust, mantle });

  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.azimuth_phi = 0;
  wave.take_off_theta = static_cast<type_real>(15.0 * 3.14159265 / 180.0);
  wave.origin_x = 0;
  wave.origin_y = 0;
  wave.origin_z = 0;
  wave.origin_time = 5;
  wave.amplitude = 1;
  wave.gaussian_half_duration = 2;

  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 256;
  window.frequency_max = 1;
  window.frequency_sampling = 2;
  window.time_window_length = 40;

  // Points: 2 in water (z < 4000), 3 in solid (z >= 4000)
  const auto ep = make_eval_points({ 500, 2000, 5000, 20000, 38000 },
                                   /*with_normals=*/true,
                                   /*with_lame=*/true);

  const auto result = fk_injection::solve(
      model, wave, window, ep, fk_injection::field_derivative::displacement);

  EXPECT_EQ(result.number_of_points(), 5);
  EXPECT_GT(result.coefficient_count(), 0);

  // Row-major (point, component, step).
  const int components = fk_injection::number_of_sample_components;
  std::vector<type_real> out(static_cast<std::size_t>(5) * components *
                             window.nstep);
  result.sample(0, window.nstep, out);

  EXPECT_TRUE(all_finite_not_all_zero(out))
      << "sampled output is NaN/Inf or all-zero";

  // The kinematic components alone must be non-zero.
  std::vector<type_real> kinematic;
  for (int ip = 0; ip < 5; ++ip)
    for (int ic = 0; ic < 3; ++ic)
      for (int is = 0; is < window.nstep; ++is)
        kinematic.push_back(out[((ip * components +
                                  fk_injection::sample_kinematic_offset + ic) *
                                 window.nstep) +
                                is]);
  EXPECT_TRUE(all_finite_not_all_zero(kinematic))
      << "sampled kinematic field is NaN/Inf or all-zero";
}

// ===========================================================================
// Test 3: ReadModelFile
// Write a small model file, read it back, check parsed fields.
// ===========================================================================
TEST(FkDriver, ReadModelFile) {
  const std::string fname = "fk_test_model.txt";

  // Write a minimal model file
  {
    std::ofstream out(fname);
    ASSERT_TRUE(out.is_open()) << "cannot open temp file for writing";
    out << "# FK test model\n";
    out << "NLAYER 2\n";
    out << "LAYER 1 2700 5800 3200 30000\n"; // crust top at z=30000 m
    out << "LAYER 2 3300 8100 4500 0\n";     // half-space at z=0
    out << "INCIDENT_WAVE p\n";
    out << "BACK_AZIMUTH 270\n"; // phi = -270 - 90 = -360 ≡ 0 deg
    out << "TAKE_OFF 30\n";
    out << "ORIGIN_WAVEFRONT 100 200 0\n";
    out << "ORIGIN_TIME 10\n";
    out << "NSTEP 512\n";
    out << "deltat 0.025\n";
    out << "FREQUENCY_MAX 2.0\n";
    out << "FREQUENCY_SAMPLING 4.0\n";
    out << "TIME_WINDOW 60\n";
    out << "AMPLITUDE 2.5\n";
  }

  auto [model, wave, window] = fk_injection::read_fk_model_file(fname);

  // Model structure
  EXPECT_EQ(model.number_of_fluid_layers(), 0);
  EXPECT_EQ(model.number_of_elastic_layers(), 2);
  EXPECT_FALSE(model.has_fluid_layer());

  // Layer properties (crust = elastic[0])
  EXPECT_NEAR(static_cast<double>(model.elastic_layers()[0].p_velocity), 5800,
              1e-3);
  EXPECT_NEAR(static_cast<double>(model.elastic_layers()[0].s_velocity), 3200,
              1e-3);
  EXPECT_NEAR(static_cast<double>(model.elastic_layers()[0].density), 2700,
              1e-3);
  // Thickness = ztop[0] - ztop[1] = 30000 - 0 = 30000
  EXPECT_NEAR(static_cast<double>(model.elastic_layers()[0].thickness), 30000,
              1e-3);

  // Wave
  EXPECT_EQ(wave.type, fk_injection::incident_wave_type::p);
  EXPECT_NEAR(static_cast<double>(wave.origin_time), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(wave.amplitude), 2.5, 1e-6);
  // Half-duration is derived from FREQUENCY_MAX (2.0 Hz): Tg = 1/2 = 0.5 s.
  EXPECT_NEAR(static_cast<double>(wave.gaussian_half_duration), 0.5, 1e-6);
  EXPECT_NEAR(static_cast<double>(wave.origin_x), 100.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(wave.origin_y), 200.0, 1e-6);
  // take_off_theta = 30 deg in radians
  EXPECT_NEAR(static_cast<double>(wave.take_off_theta),
              30.0 * 3.14159265358979 / 180.0, 1e-5);

  // Window
  EXPECT_EQ(window.nstep, 512);
  EXPECT_NEAR(static_cast<double>(window.dt), 0.025, 1e-7);
  EXPECT_NEAR(static_cast<double>(window.frequency_max), 2.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(window.frequency_sampling), 4.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(window.time_window_length), 60.0, 1e-6);
}

// ===========================================================================
// check_eval_points: every array agrees, and the traction inputs come together.
// ===========================================================================
static void expect_rejected(const fk_injection::EvalPoints &points,
                            const std::string &reason) {
  try {
    fk_injection::impl::check_eval_points(points);
    FAIL() << "expected a rejection mentioning '" << reason << "'";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(reason), std::string::npos)
        << error.what();
  }
}

TEST(FkDriver, EvalPointsAccepted) {
  EXPECT_NO_THROW(
      fk_injection::impl::check_eval_points(make_eval_points({ 0, 1000 })));
  EXPECT_NO_THROW(fk_injection::impl::check_eval_points(make_eval_points(
      { 0, 1000 }, /*with_normals=*/true, /*with_lame=*/true)));

  auto with_medium = make_eval_points({ 0, 1000 });
  with_medium.medium = { specfem::plugin::Medium::elastic,
                         specfem::plugin::Medium::elastic };
  EXPECT_NO_THROW(fk_injection::impl::check_eval_points(with_medium));
}

TEST(FkDriver, EvalPointsRejectSizeMismatch) {
  auto short_y = make_eval_points({ 0, 1000 });
  short_y.y.pop_back();
  expect_rejected(short_y, "y has 1 entries, expected 2");

  auto short_z = make_eval_points({ 0, 1000 });
  short_z.z.pop_back();
  expect_rejected(short_z, "z has 1 entries, expected 2");

  auto short_medium = make_eval_points({ 0, 1000 });
  short_medium.medium = { specfem::plugin::Medium::elastic };
  expect_rejected(short_medium, "medium has 1 entries, expected 2");
}

TEST(FkDriver, EvalPointsRejectPartialTractionInputs) {
  expect_rejected(make_eval_points({ 0, 1000 }, /*with_normals=*/true,
                                   /*with_lame=*/false),
                  "lame_ratio_xi1 has 0 entries, expected 2");
  expect_rejected(make_eval_points({ 0, 1000 }, /*with_normals=*/false,
                                   /*with_lame=*/true),
                  "normal_x has 0 entries, expected 2");
}

// ===========================================================================
// check_supported_slowness: 0 < p < 1/v for every layer velocity.
// ===========================================================================
static fk_injection::LayeredModel make_slowness_model() {
  fk_injection::AcousticLayer ocean;
  ocean.p_velocity = 1500;
  ocean.density = 1025;
  ocean.thickness = 4000;

  fk_injection::ElasticIsotropicLayer mantle;
  mantle.p_velocity = 8100;
  mantle.s_velocity = 4500;
  mantle.density = 3300;

  return fk_injection::LayeredModel({ ocean }, { mantle });
}

static void expect_unsupported(double p, const std::string &reason) {
  try {
    fk_injection::impl::check_supported_slowness(make_slowness_model(), p);
    FAIL() << "expected p = " << p << " to be rejected";
  } catch (const specfem::plugin::Unsupported &error) {
    EXPECT_NE(std::string(error.what()).find(reason), std::string::npos)
        << error.what();
  }
}

TEST(FkDriver, SlownessAccepted) {
  EXPECT_NO_THROW(fk_injection::impl::check_supported_slowness(
      make_slowness_model(), 0.5 / 8100.0));
  // The fluid's slower velocity doesn't tighten the limit.
  EXPECT_NO_THROW(fk_injection::impl::check_supported_slowness(
      make_slowness_model(), 0.999 / 8100.0));
}

TEST(FkDriver, SlownessRejected) {
  expect_unsupported(0.0, "vertical incidence");
  expect_unsupported(-1.0e-5, "vertical incidence");
  // Exactly critical: 0/0 in the propagators.
  expect_unsupported(1.0 / 8100.0, "critical");
  expect_unsupported(1.5 / 8100.0, "critical");
  expect_unsupported(std::nan(""), "vertical incidence");
}

TEST(FkDriver, SolveRejectsVerticalIncidence) {
  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.take_off_theta = 0;

  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 256;
  window.frequency_max = 1;
  window.frequency_sampling = 2;
  window.time_window_length = 40;

  EXPECT_THROW(fk_injection::solve(make_slowness_model(), wave, window,
                                   make_eval_points({ 5000 }),
                                   fk_injection::field_derivative::velocity),
               specfem::plugin::Unsupported);
}

// ===========================================================================
// ReferenceConvention::specfem3d: amplitudes N_P times exact, and delays
// stretched by N/(N-1).
// ===========================================================================
TEST(FkDriver, Specfem3dConventionScalesAmplitudeAndDelay) {
  const auto model = make_slowness_model();

  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.take_off_theta = static_cast<type_real>(20.0 * 3.14159265 / 180.0);
  wave.origin_time = 5;
  wave.gaussian_half_duration = 2;

  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 256;
  window.frequency_max = 1;
  window.frequency_sampling = 10; // dt_min = 0.1, N_P = 2
  window.time_window_length = 60;

  // The first point has no delay; the second has one.
  auto points = make_eval_points({ 6000, 6000 });
  points.x[1] = 10000;

  const auto exact = fk_injection::solve(
      model, wave, window, points, fk_injection::field_derivative::displacement,
      fk_injection::ReferenceConvention::exact);
  const auto specfem3d = fk_injection::solve(
      model, wave, window, points, fk_injection::field_derivative::displacement,
      fk_injection::ReferenceConvention::specfem3d);
  ASSERT_EQ(exact.resampling_rate(), 2);

  const int components = fk_injection::number_of_sample_components;
  const std::size_t size =
      static_cast<std::size_t>(2) * components * window.nstep;
  std::vector<type_real> out_exact(size);
  std::vector<type_real> out_specfem3d(size);
  exact.sample(0, window.nstep, out_exact);
  specfem3d.sample(0, window.nstep, out_specfem3d);

  // Point 0, all components and steps.
  const std::size_t first_point =
      static_cast<std::size_t>(components) * window.nstep;
  double peak = 0.0;
  for (std::size_t i = 0; i < first_point; ++i) {
    peak = std::max(peak, std::fabs(static_cast<double>(out_exact[i])));
  }
  ASSERT_GT(peak, 0.0);
  for (std::size_t i = 0; i < first_point; ++i) {
    EXPECT_NEAR(static_cast<double>(out_specfem3d[i]),
                2.0 * static_cast<double>(out_exact[i]), 1.0e-5 * peak)
        << "entry " << i;
  }

  // Offset = pad - tau*stretch/dt_min, and the first point's offset is the pad.
  const double exact_delay =
      exact.points()[0].offset - exact.points()[1].offset;
  const double stretched_delay =
      specfem3d.points()[0].offset - specfem3d.points()[1].offset;
  const double span = wave.ray_parameter(model) * 10000.0;
  const int fft_samples =
      fk_injection::compute_fk_sizes(window, span).number_of_fft_samples;
  EXPECT_NEAR(stretched_delay / exact_delay,
              static_cast<double>(fft_samples) /
                  static_cast<double>(fft_samples - 1),
              1.0e-9);
}
