#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/injection_provider.hpp"
#include "specfem/runtime_configuration/injection.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>

// ---------------------------------------------------------------------------
// Helper: build a minimal 1-point EvalPoints (no traction views).
// ---------------------------------------------------------------------------
static specfem::injection::fk::EvalPoints make_one_point_eval_points() {
  specfem::injection::fk::EvalPoints ep;
  ep.x = Kokkos::View<type_real *>("x", 1);
  ep.y = Kokkos::View<type_real *>("y", 1);
  ep.z = Kokkos::View<type_real *>("z", 1);
  return ep;
}

// ---------------------------------------------------------------------------
// Inline YAML helper string: ocean + crust + half-space with incidence and
// time-window blocks.
// ---------------------------------------------------------------------------
static const char *const k_inline_yaml = R"yaml(
method: fk
layers:
  - acoustic: { rho: 1025, vp: 1500, thickness: 2000 }
  - elastic:  { rho: 2800, vp: 6000, vs: 3500, thickness: 35000 }
  - elastic:  { rho: 3300, vp: 8100, vs: 4500, thickness: 0 }
incidence:
  type: P
  back-azimuth: 30.0
  take-off: 25.0
  origin: [0, 0, 0]
  origin-time: 0.0
  amplitude: 1.0
  half-duration: 0.0
time-window:
  frequency-max: 1.0
  frequency-sampling: 10.0
  length: 128.0
)yaml";

// ===========================================================================
// TEST 1: InlineLayersParse
// ===========================================================================
TEST(InjectionConfig, InlineLayersParse) {
  const YAML::Node node = YAML::Load(k_inline_yaml);
  const specfem::runtime_configuration::Injection cfg(node);

  EXPECT_TRUE(cfg.is_enabled());
  EXPECT_EQ(cfg.method(), "fk");

  const auto model = cfg.to_layered_model();
  EXPECT_EQ(model.number_of_fluid_layers(), 1);
  EXPECT_EQ(model.number_of_elastic_layers(), 2);

  const auto wave = cfg.to_incident_wave();
  EXPECT_EQ(wave.type, specfem::injection::fk::incident_wave_type::p);

  // back-azimuth 30 => phi_deg = -30 - 90 = -120 degrees
  const double expected_phi = (-30.0 - 90.0) * (3.141592653589793 / 180.0);
  EXPECT_NEAR(static_cast<double>(wave.azimuth_phi), expected_phi, 1.0e-5);

  // take-off 25 degrees
  const double expected_theta = 25.0 * (3.141592653589793 / 180.0);
  EXPECT_NEAR(static_cast<double>(wave.take_off_theta), expected_theta, 1.0e-5);

  // type_real may be single precision; use a float-round-trip tolerance.
  const auto window = cfg.to_time_window(static_cast<type_real>(0.05), 256);
  EXPECT_NEAR(static_cast<double>(window.dt), 0.05, 1.0e-6);
  EXPECT_EQ(window.nstep, 256);
  EXPECT_NEAR(static_cast<double>(window.frequency_max), 1.0, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(window.frequency_sampling), 10.0, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(window.time_window_length), 128.0, 1.0e-6);
}

// ===========================================================================
// TEST 2: InstantiateReturnsFkProvider
// ===========================================================================
TEST(InjectionConfig, InstantiateReturnsFkProvider) {
  const YAML::Node node = YAML::Load(k_inline_yaml);
  const specfem::runtime_configuration::Injection cfg(node);

  const auto ep = make_one_point_eval_points();
  const auto provider = cfg.instantiate<specfem::element::dimension_tag::dim3>(
      ep, static_cast<type_real>(0.05), 256, /*compute_traction=*/false);

  EXPECT_NE(provider, nullptr);
}

// ===========================================================================
// TEST 3: AzimuthConvention
// ===========================================================================
TEST(InjectionConfig, AzimuthConvention) {
  const char *yaml_str = R"yaml(
layers:
  - elastic: { rho: 2800, vp: 6000, vs: 3500, thickness: 35000 }
  - elastic: { rho: 3300, vp: 8100, vs: 4500, thickness: 0 }
incidence:
  type: P
  azimuth: 40.0
  take-off: 20.0
time-window:
  frequency-max: 1.0
  frequency-sampling: 10.0
  length: 128.0
)yaml";

  const YAML::Node node = YAML::Load(yaml_str);
  const specfem::runtime_configuration::Injection cfg(node);

  const auto wave = cfg.to_incident_wave();

  // azimuth 40 => phi_deg = 90 - 40 = 50 degrees
  const double expected_phi = (90.0 - 40.0) * (3.141592653589793 / 180.0);
  EXPECT_NEAR(static_cast<double>(wave.azimuth_phi), expected_phi, 1.0e-5);
}

// ===========================================================================
// TEST 4: RejectsBothLayersAndModelFile
// ===========================================================================
TEST(InjectionConfig, RejectsBothLayersAndModelFile) {
  const char *yaml_str = R"yaml(
layers:
  - elastic: { rho: 2800, vp: 6000, vs: 3500, thickness: 0 }
model-file: some_file.dat
incidence:
  type: P
  take-off: 20.0
time-window:
  frequency-max: 1.0
  frequency-sampling: 10.0
  length: 128.0
)yaml";

  const YAML::Node node = YAML::Load(yaml_str);
  EXPECT_THROW(specfem::runtime_configuration::Injection cfg(node),
               std::runtime_error);
}

// ===========================================================================
// TEST 5: RejectsNeitherLayersNorModelFile
// ===========================================================================
TEST(InjectionConfig, RejectsNeitherLayersNorModelFile) {
  const char *yaml_str = R"yaml(
method: fk
enabled: true
)yaml";

  const YAML::Node node = YAML::Load(yaml_str);
  EXPECT_THROW(specfem::runtime_configuration::Injection cfg(node),
               std::runtime_error);
}

// ===========================================================================
// TEST 6: RejectsAcousticWithVs
// ===========================================================================
TEST(InjectionConfig, RejectsAcousticWithVs) {
  const char *yaml_str = R"yaml(
layers:
  - acoustic: { rho: 1025, vp: 1500, vs: 0, thickness: 2000 }
  - elastic:  { rho: 2800, vp: 6000, vs: 3500, thickness: 0 }
incidence:
  type: P
  take-off: 20.0
time-window:
  frequency-max: 1.0
  frequency-sampling: 10.0
  length: 128.0
)yaml";

  const YAML::Node node = YAML::Load(yaml_str);
  EXPECT_THROW(specfem::runtime_configuration::Injection cfg(node),
               std::runtime_error);
}

// ===========================================================================
// TEST 7: RejectsUnknownMethod
// Construction succeeds; instantiate() throws.
// ===========================================================================
TEST(InjectionConfig, RejectsUnknownMethod) {
  const char *yaml_str = R"yaml(
method: axisem
layers:
  - elastic: { rho: 2800, vp: 6000, vs: 3500, thickness: 35000 }
  - elastic: { rho: 3300, vp: 8100, vs: 4500, thickness: 0 }
incidence:
  type: P
  take-off: 20.0
time-window:
  frequency-max: 1.0
  frequency-sampling: 10.0
  length: 128.0
)yaml";

  const YAML::Node node = YAML::Load(yaml_str);
  const specfem::runtime_configuration::Injection cfg(node);
  EXPECT_EQ(cfg.method(), "axisem");

  const auto ep = make_one_point_eval_points();
  EXPECT_THROW((cfg.instantiate<specfem::element::dimension_tag::dim3>(
                   ep, static_cast<type_real>(0.05), 256, false)),
               std::runtime_error);
}
