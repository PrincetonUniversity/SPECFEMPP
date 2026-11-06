#include "field_helpers.hpp"
#include "fk_injection/fk_injection.hpp"
#include "plane_wave_response.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

using fk_injection::type_real;

// ---------------------------------------------------------------------------
// Points at x = y = 0 and the given depths. With traction, the normal points
// in +z.
// ---------------------------------------------------------------------------
static std::vector<fk_injection::FieldPoint>
make_points(const std::vector<double> &depths, bool with_traction = false) {
  std::vector<fk_injection::FieldPoint> points(depths.size());
  for (std::size_t i = 0; i < depths.size(); ++i) {
    points[i].depth = static_cast<type_real>(depths[i]);
    if (with_traction) {
      points[i].normal_z = 1;
    }
  }
  return points;
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
TEST(FkField, ElasticHalfSpaceRunsFinite) {
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

  fk_injection::LayeredModel model({ crust, mantle });

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
  window.frequency_sampling = 2;
  window.time_window_length = 40;

  // Points at various depths in the crust and just below the surface.
  const auto ep = make_points({ 0, 5000, 15000, 30000, 40000 });

  const auto result =
      fk_injection_tests_impl::field_at(model, wave, window, ep);

  // Shape checks
  EXPECT_EQ(result.number_of_points(), 5);
  EXPECT_GT(result.series().coefficient_count(), 0);

  std::vector<type_real> out;
  for (const auto quantity : fk_injection::ElasticMedium::quantities) {
    const auto values =
        fk_injection_tests_impl::sample(result, quantity, 0, window.nstep);
    out.insert(out.end(), values.begin(), values.end());
  }
  EXPECT_TRUE(all_finite_not_all_zero(out))
      << "sampled output is NaN/Inf or all-zero";
}

// ===========================================================================
// Test 2: AcousticElasticRunsFinite
// Ocean (fluid) + crust + mantle, incident P.  Points in water AND solid.
// ===========================================================================
TEST(FkField, AcousticElasticRunsFinite) {
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

  fk_injection::LayeredModel model({ ocean, crust, mantle });

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
  window.frequency_sampling = 2;
  window.time_window_length = 40;

  // Points: 2 in water (z < 4000), 3 in solid (z >= 4000)
  const auto water = make_points({ 500, 2000 }, /*with_traction=*/true);
  const auto solid = make_points({ 5000, 20000, 38000 },
                                 /*with_traction=*/true);
  const auto box = fk_injection_tests_impl::bounding_box(water);
  const auto acoustic =
      fk_injection_tests_impl::field_at<fk_injection::AcousticMedium>(
          model, wave, window, box, water);
  const auto elastic =
      fk_injection_tests_impl::field_at(model, wave, window, box, solid);

  EXPECT_EQ(acoustic.number_of_points(), 2);
  EXPECT_EQ(elastic.number_of_points(), 3);
  EXPECT_GT(acoustic.series().coefficient_count(), 0);

  for (const auto quantity : fk_injection::AcousticMedium::quantities)
    EXPECT_TRUE(all_finite_not_all_zero(
        fk_injection_tests_impl::sample(acoustic, quantity, 0, window.nstep)))
        << "acoustic quantity " << static_cast<int>(quantity)
        << " is NaN/Inf or all-zero";
  for (const auto quantity : fk_injection::ElasticMedium::quantities)
    EXPECT_TRUE(all_finite_not_all_zero(
        fk_injection_tests_impl::sample(elastic, quantity, 0, window.nstep)))
        << "elastic quantity " << static_cast<int>(quantity)
        << " is NaN/Inf or all-zero";
}

// ===========================================================================
// Test 3: ReadModelFile
// Write a small model file, read it back, check parsed fields.
// ===========================================================================
TEST(FkField, ReadModelFile) {
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

  auto [model, wave, window, frequency_max, has_origin] =
      fk_injection::read_fk_model_file(fname);
  EXPECT_TRUE(has_origin);

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
  EXPECT_NEAR(static_cast<double>(wave.origin_x), 100.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(wave.origin_y), 200.0, 1e-6);
  // take_off_theta = 30 deg in radians
  EXPECT_NEAR(static_cast<double>(wave.take_off_theta),
              30.0 * 3.14159265358979 / 180.0, 1e-5);

  // Window
  EXPECT_EQ(window.nstep, 512);
  EXPECT_NEAR(static_cast<double>(window.dt), 0.025, 1e-7);
  EXPECT_NEAR(static_cast<double>(frequency_max), 2.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(window.frequency_sampling), 4.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(window.time_window_length), 60.0, 1e-6);
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

  return fk_injection::LayeredModel({ ocean, mantle });
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

TEST(FkField, SlownessAccepted) {
  EXPECT_NO_THROW(fk_injection::impl::check_supported_slowness(
      make_slowness_model(), 0.5 / 8100.0));
  // The fluid's slower velocity doesn't tighten the limit.
  EXPECT_NO_THROW(fk_injection::impl::check_supported_slowness(
      make_slowness_model(), 0.999 / 8100.0));
}

TEST(FkField, SlownessRejected) {
  expect_unsupported(0.0, "vertical incidence");
  expect_unsupported(-1.0e-5, "vertical incidence");
  // Exactly critical: 0/0 in the propagators.
  expect_unsupported(1.0 / 8100.0, "critical");
  expect_unsupported(1.5 / 8100.0, "critical");
  expect_unsupported(std::nan(""), "vertical incidence");
}

TEST(FkField, ResponseRejectsVerticalIncidence) {
  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.take_off_theta = 0;

  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 256;
  window.frequency_sampling = 2;
  window.time_window_length = 40;

  EXPECT_THROW(
      fk_injection::PlaneWaveResponse(make_slowness_model(), wave, window, {}),
      specfem::plugin::Unsupported);
}

// ===========================================================================
// ReferenceConvention::specfem3d: amplitudes N_P times exact, and delays
// stretched by N/(N-1).
// ===========================================================================
TEST(FkField, Specfem3dConventionScalesAmplitudeAndDelay) {
  // No fluid: only the exact convention lengthens the window for a fluid's
  // reverberation, which would make the two windows differ.
  const fk_injection::LayeredModel model(
      { make_slowness_model().elastic_layers().back() });

  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.take_off_theta = static_cast<type_real>(20.0 * 3.14159265 / 180.0);
  wave.gaussian_half_duration = 2;
  // The half-space top, so a point at x = y = 0 has no delay.
  wave.origin_z = static_cast<type_real>(-model.halfspace_top_depth());

  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 256;
  window.frequency_sampling = 10; // dt_min = 0.1, N_P = 2
  window.time_window_length = 60;

  // The first point has no delay; the second has one.
  auto points = make_points({ 6000, 6000 });
  points[1].x = 10000;

  const auto exact = fk_injection_tests_impl::field_at(
      model, wave, window, points, fk_injection::ReferenceConvention::exact);
  const auto specfem3d = fk_injection_tests_impl::field_at(
      model, wave, window, points,
      fk_injection::ReferenceConvention::specfem3d);
  ASSERT_EQ(exact.resampling_rate(), 2);

  // Point 0, all components and steps. The two also differ by the N/(N-1)
  // stretch of their frequency grids, so compare peaks, not samples; the
  // stretch also moves each peak between samples, by up to 2e-3 here.
  const std::size_t first_point = static_cast<std::size_t>(3) * window.nstep;
  double peak_exact = 0.0;
  double peak_specfem3d = 0.0;
  for (const auto quantity : fk_injection::ElasticMedium::quantities) {
    const auto out_exact =
        fk_injection_tests_impl::sample(exact, quantity, 0, window.nstep);
    const auto out_specfem3d =
        fk_injection_tests_impl::sample(specfem3d, quantity, 0, window.nstep);
    for (std::size_t i = 0; i < first_point; ++i) {
      peak_exact =
          std::max(peak_exact, std::fabs(static_cast<double>(out_exact[i])));
      peak_specfem3d = std::max(
          peak_specfem3d, std::fabs(static_cast<double>(out_specfem3d[i])));
    }
  }
  ASSERT_GT(peak_exact, 0.0);
  EXPECT_NEAR(peak_specfem3d / peak_exact, 2.0, 2.0e-3);

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

// ===========================================================================
// A homogeneous half-space under a P wave with a Gaussian source. The origin
// sits below the evaluation points, so the first arrival there is the upgoing
// incident wave.
// ===========================================================================
namespace field_tests_impl {

constexpr double density = 2700.0;
constexpr double vp = 3000.0;
constexpr double vs = 1732.051;
constexpr double origin_z = -223944.719;
constexpr double take_off = 20.0 * std::numbers::pi / 180.0;

fk_injection::ElasticIsotropicLayer homogeneous_layer(double thickness) {
  fk_injection::ElasticIsotropicLayer layer;
  layer.density = static_cast<type_real>(density);
  layer.p_velocity = static_cast<type_real>(vp);
  layer.s_velocity = static_cast<type_real>(vs);
  layer.thickness = static_cast<type_real>(thickness);
  return layer;
}

fk_injection::IncidentWave gaussian_p_wave() {
  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.take_off_theta = static_cast<type_real>(take_off);
  wave.azimuth_phi = static_cast<type_real>(-90.0 * std::numbers::pi / 180.0);
  wave.origin_z = static_cast<type_real>(origin_z);
  wave.gaussian_half_duration = 5;
  return wave;
}

fk_injection::TimeWindow long_window(double frequency_sampling) {
  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.15);
  window.nstep = 1200;
  window.frequency_sampling = static_cast<type_real>(frequency_sampling);
  window.time_window_length = 180;
  return window;
}

/** @brief Evaluation points at x = y = 0 with a downward normal. */
std::vector<fk_injection::FieldPoint>
points_with_traction(const std::vector<double> &depths) {
  auto points = make_points(depths, /*with_traction=*/true);
  for (auto &point : points)
    point.normal_z = -1;
  return points;
}

/** @brief One sampled quantity with (point, component, step) access. */
struct SampledField {
  std::vector<type_real> values{};
  int number_of_components = 0;
  int number_of_steps = 0;

  SampledField(
      const fk_injection::FieldSampler<fk_injection::ElasticMedium> &result,
      specfem::plugin::Quantity quantity, int steps)
      : values(fk_injection_tests_impl::sample(result, quantity, 0, steps)),
        number_of_components(fk_injection::number_of_components(quantity)),
        number_of_steps(steps) {}

  double operator()(int point, int component, int step) const {
    return static_cast<double>(
        values[(static_cast<std::size_t>(point) * number_of_components +
                component) *
                   number_of_steps +
               step]);
  }
};

constexpr int y_component = 1;
constexpr int z_component = 2;

/** @brief Step at which |component| peaks for one point. */
int peak_step(const SampledField &field, int point, int component) {
  int step_at_peak = 0;
  double peak = 0.0;
  for (int step = 0; step < field.number_of_steps; ++step) {
    const double magnitude = std::fabs(field(point, component, step));
    if (magnitude > peak) {
      peak = magnitude;
      step_at_peak = step;
    }
  }
  return step_at_peak;
}

} // namespace field_tests_impl

// Traction and velocity of an upgoing plane P wave satisfy the impedance
// relations; a sign or scale error in either breaks them.
TEST(FkField, TractionConsistentWithImpedance) {
  const fk_injection::LayeredModel model(
      { field_tests_impl::homogeneous_layer(0) });
  const auto window = field_tests_impl::long_window(2.0);
  const auto result = fk_injection_tests_impl::field_at(
      model, field_tests_impl::gaussian_p_wave(), window,
      field_tests_impl::points_with_traction({ 120000.0 }));
  const field_tests_impl::SampledField velocity(
      result, specfem::plugin::Quantity::velocity, window.nstep);
  const field_tests_impl::SampledField traction(
      result, specfem::plugin::Quantity::traction, window.nstep);

  const int step =
      field_tests_impl::peak_step(velocity, 0, field_tests_impl::z_component);
  ASSERT_GT(std::fabs(velocity(0, field_tests_impl::z_component, step)), 0.0);

  const double mu =
      field_tests_impl::density * field_tests_impl::vs * field_tests_impl::vs;
  const double lambda =
      field_tests_impl::density * field_tests_impl::vp * field_tests_impl::vp -
      2.0 * mu;
  const double cosine = std::cos(field_tests_impl::take_off);

  EXPECT_NEAR(traction(0, field_tests_impl::z_component, step) /
                  (field_tests_impl::density * field_tests_impl::vp *
                   velocity(0, field_tests_impl::z_component, step)),
              (lambda + 2.0 * mu * cosine * cosine) /
                  ((lambda + 2.0 * mu) * cosine),
              0.15);
  EXPECT_NEAR(traction(0, field_tests_impl::y_component, step) /
                  (field_tests_impl::density * field_tests_impl::vs *
                   velocity(0, field_tests_impl::y_component, step)),
              2.0 * (field_tests_impl::vs / field_tests_impl::vp) * cosine,
              0.20);
}

// A delayed P pulse at (x = y = 0, SEM z) arrives at eta_p * (z - z0).
TEST(FkField, DelayedArrivalMatchesPlaneWaveTiming) {
  const fk_injection::LayeredModel model(
      { field_tests_impl::homogeneous_layer(0) });
  const auto window = field_tests_impl::long_window(10.0);
  const std::vector<double> depths = { 50005.0, 20000.0 };
  const auto result = fk_injection_tests_impl::field_at(
      model, field_tests_impl::gaussian_p_wave(), window, make_points(depths));
  const field_tests_impl::SampledField field(
      result, specfem::plugin::Quantity::velocity, window.nstep);

  const double p = std::sin(field_tests_impl::take_off) / field_tests_impl::vp;
  const double eta =
      std::sqrt(1.0 / (field_tests_impl::vp * field_tests_impl::vp) - p * p);
  const int velocity_z = field_tests_impl::z_component;
  for (int point = 0; point < 2; ++point) {
    // Velocity of a Gaussian peaks at its inflection, up to a half-duration of
    // 5 s before the displacement center.
    EXPECT_NEAR(field_tests_impl::peak_step(field, point, velocity_z) *
                    static_cast<double>(window.dt),
                eta * (-depths[point] - field_tests_impl::origin_z), 6.0)
        << "depth " << depths[point];
  }
}

// The series are periodic, so an origin time one FFT period away would give the
// undelayed field; a third of a period away is still fine.
TEST(FkField, RejectsFkTimesOutsideTheFftPeriod) {
  const fk_injection::LayeredModel model(
      { field_tests_impl::homogeneous_layer(0) });
  const auto window = field_tests_impl::long_window(2.0);
  const auto points = make_points({ 20000.0 });
  const auto sizes = fk_injection::compute_fk_sizes(window, 0.0);
  const double period = static_cast<double>(sizes.number_of_fft_samples) *
                        static_cast<double>(sizes.resampling_rate) *
                        static_cast<double>(window.dt);

  auto wave = field_tests_impl::gaussian_p_wave();
  wave.origin_time = static_cast<type_real>(-period / 8.0);
  EXPECT_NO_THROW(
      fk_injection_tests_impl::field_at(model, wave, window, points));

  for (const double origin_time : { -period, period }) {
    wave.origin_time = static_cast<type_real>(origin_time);
    try {
      fk_injection_tests_impl::field_at(model, wave, window, points);
      FAIL() << "expected a rejection for origin time " << origin_time;
    } catch (const std::runtime_error &error) {
      EXPECT_NE(std::string(error.what()).find("FFT period"), std::string::npos)
          << error.what();
    }
  }
}

// In a half-space at depth d, the incident and reflected P pulses are centered
// at tau - eta d and tau + eta d, where the velocity crosses zero. A frequency
// grid that doesn't match the FFT moves them by eta d / N.
TEST(FkField, ReflectedArrivalsMatchPlaneWaveTiming) {
  const fk_injection::LayeredModel model(
      { field_tests_impl::homogeneous_layer(0) });
  const double depth = 50000.0;
  const double delay = 25.0;
  const double p = std::sin(field_tests_impl::take_off) / field_tests_impl::vp;
  const double eta =
      std::sqrt(1.0 / (field_tests_impl::vp * field_tests_impl::vp) - p * p);

  auto wave = field_tests_impl::gaussian_p_wave();
  wave.gaussian_half_duration = 1;
  wave.origin_z = static_cast<type_real>(-delay / eta);

  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.1);
  window.nstep = 500;
  window.frequency_sampling = 10;
  window.time_window_length = 50;

  const field_tests_impl::SampledField field(
      fk_injection_tests_impl::field_at(model, wave, window,
                                        make_points({ depth })),
      specfem::plugin::Quantity::velocity, window.nstep);
  const int velocity_z = field_tests_impl::z_component;

  const auto zero_crossing_near = [&](const double expected) {
    const int first = static_cast<int>(expected / window.dt) - 20;
    for (int step = first; step < first + 40; ++step) {
      const double before = field(0, velocity_z, step);
      const double after = field(0, velocity_z, step + 1);
      if (before * after < 0.0 && std::fabs(before - after) > 0.0)
        return (step + before / (before - after)) * window.dt;
    }
    return -1.0;
  };

  for (const double expected : { delay - eta * depth, delay + eta * depth })
    EXPECT_NEAR(zero_crossing_near(expected), expected, 3.0e-3);
}

// origin_time is the FK time at step 0, so origin_time = -k dt delays the field
// by exactly k steps, whether or not k dt is a multiple of dt_min, and nothing
// wraps around into the steps before the arrival.
TEST(FkField, OriginTimeDelaysTheField) {
  const fk_injection::LayeredModel model(
      { field_tests_impl::homogeneous_layer(0) });
  // N_P = 3, so 80 steps is 26 2/3 resampled samples.
  const auto window = field_tests_impl::long_window(2.0);
  const auto points = make_points({ 50005.0, 20000.0 });
  const int velocity_z = field_tests_impl::z_component;

  const field_tests_impl::SampledField reference(
      fk_injection_tests_impl::field_at(
          model, field_tests_impl::gaussian_p_wave(), window, points),
      specfem::plugin::Quantity::velocity, window.nstep);

  for (const int delay_steps : { 80, 400 }) {
    auto wave = field_tests_impl::gaussian_p_wave();
    wave.origin_time =
        static_cast<type_real>(-delay_steps * static_cast<double>(window.dt));
    const field_tests_impl::SampledField delayed(
        fk_injection_tests_impl::field_at(model, wave, window, points),
        specfem::plugin::Quantity::velocity, window.nstep);

    for (int point = 0; point < 2; ++point) {
      double peak = 0.0;
      double difference = 0.0;
      for (int step = 0; step + delay_steps < window.nstep; ++step) {
        const double value = reference(point, velocity_z, step);
        peak = std::max(peak, std::fabs(value));
        difference = std::max(
            difference,
            std::fabs(delayed(point, velocity_z, step + delay_steps) - value));
      }
      ASSERT_GT(peak, 0.0);
      EXPECT_LT(difference / peak, 5.0e-3)
          << "delay " << delay_steps << ", point " << point;

      double before_arrival = 0.0;
      for (int step = 0; step < delay_steps; ++step)
        before_arrival = std::max(before_arrival,
                                  std::fabs(delayed(point, velocity_z, step)));
      EXPECT_LT(before_arrival / peak, 5.0e-3)
          << "delay " << delay_steps << ", point " << point;
    }
  }
}

// Splitting a half-space into two identical layers changes nothing physically,
// so the field must not change. It routes evaluation through the finite-layer
// propagator and moves the half-space top, which anchors the time zero.
TEST(FkField, IdenticalLayerSplitReducesToHalfspace) {
  const fk_injection::LayeredModel halfspace(
      { field_tests_impl::homogeneous_layer(0) });
  const fk_injection::LayeredModel split(
      { field_tests_impl::homogeneous_layer(30000.0),
        field_tests_impl::homogeneous_layer(0) });
  const auto wave = field_tests_impl::gaussian_p_wave();
  const auto window = field_tests_impl::long_window(10.0);
  const std::vector<double> depths = { 5000.0,  15000.0, 29900.0,
                                       30100.0, 60000.0, 120000.0 };
  const auto points = field_tests_impl::points_with_traction(depths);

  const auto reference_sampler =
      fk_injection_tests_impl::field_at(halfspace, wave, window, points);
  const auto reduced_sampler =
      fk_injection_tests_impl::field_at(split, wave, window, points);

  for (const auto quantity : fk_injection::ElasticMedium::quantities) {
    const field_tests_impl::SampledField reference(reference_sampler, quantity,
                                                   window.nstep);
    const field_tests_impl::SampledField reduced(reduced_sampler, quantity,
                                                 window.nstep);
    for (std::size_t point = 0; point < depths.size(); ++point) {
      double peak = 0.0;
      double difference = 0.0;
      for (int component = 0; component < 3; ++component) {
        for (int step = 0; step < window.nstep; ++step) {
          const double value = reference(point, component, step);
          peak = std::max(peak, std::fabs(value));
          difference = std::max(
              difference, std::fabs(reduced(point, component, step) - value));
        }
      }
      ASSERT_GT(peak, 0.0);
      // Limited by single-precision coefficient storage, about 1e-3 of peak.
      EXPECT_LT(difference / peak, 5.0e-3)
          << "depth " << depths[point] << ", quantity "
          << static_cast<int>(quantity);
    }
  }
}
