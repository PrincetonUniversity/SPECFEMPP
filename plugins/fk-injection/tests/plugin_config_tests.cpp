#include "plugin_config.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fk_injection_test {

constexpr double degree = std::numbers::pi / 180.0;

// Ocean + crust + half-space.
constexpr const char *inline_config = R"yaml(
layers:
  - acoustic: { rho: 1025, vp: 1500, thickness: 2000 }
  - elastic:  { rho: 2800, vp: 6000, vs: 3500, thickness: 35000 }
  - elastic:  { rho: 3300, vp: 8100, vs: 4500, thickness: 0 }
incidence:
  type: P
  back-azimuth: 30.0
  take-off: 25.0
  origin: [1, 2, 3]
  amplitude: 2.0
time-window:
  frequency-max: 1.0
  frequency-sampling: 10.0
  length: 128.0
)yaml";

constexpr const char *valid_incidence =
    "{ type: P, azimuth: 40, take-off: 20, origin: [0, 0, 0] }";
constexpr const char *valid_window =
    "{ frequency-max: 1.0, frequency-sampling: 10.0, length: 128.0 }";

// A two-layer document; an empty block is left out.
std::string config_text(const std::string &incidence = valid_incidence,
                        const std::string &window = valid_window) {
  std::string text = R"yaml(
layers:
  - elastic: { rho: 2800, vp: 6000, vs: 3500, thickness: 35000 }
  - elastic: { rho: 3300, vp: 8100, vs: 4500, thickness: 0 }
)yaml";
  if (!incidence.empty())
    text += "incidence: " + incidence + "\n";
  if (!window.empty())
    text += "time-window: " + window + "\n";
  return text;
}

// A model file in a temporary directory, removed with the object.
class ModelFile {
public:
  ModelFile(const std::string &name, const std::string &contents)
      : path_(std::filesystem::temp_directory_path() / name) {
    std::ofstream(path_) << contents;
  }
  ~ModelFile() { std::filesystem::remove(path_); }
  std::string path() const { return path_.string(); }

private:
  std::filesystem::path path_;
};

constexpr const char *model_file = R"(NLAYER 2
LAYER 1 2500. 2500. 1250.000 0.
LAYER 2 2700. 3000. 1732.051 -30000.
INCIDENT_WAVE p
BACK_AZIMUTH 0.
TAKE_OFF 20.
ORIGIN_WAVEFRONT 0. 0. -223944.719
FREQUENCY_MAX 0.2
TIME_WINDOW 180.
)";

struct RejectionCase {
  std::string name;
  std::string document;
  std::string reason; ///< Part of the error message, usually the offending key
};

std::vector<RejectionCase> rejection_cases() {
  return {
    { "BothAzimuthConventions",
      config_text("{ type: P, azimuth: 40, back-azimuth: 0, take-off: 20, "
                  "origin: [0, 0, 0] }"),
      "back-azimuth or azimuth" },
    { "BothLayersAndModelFile",
      R"yaml(
layers:
  - elastic: { rho: 2800, vp: 6000, vs: 3500, thickness: 0 }
model-file: some_file.dat
)yaml",
      "exactly one of layers or model-file" },
    { "NeitherLayersNorModelFile", "material-tolerance: 1.0e-3\n",
      "exactly one of layers or model-file" },
    { "AcousticWithVs",
      R"yaml(
layers:
  - acoustic: { rho: 1025, vp: 1500, vs: 0, thickness: 2000 }
  - elastic:  { rho: 2800, vp: 6000, vs: 3500, thickness: 0 }
)yaml",
      "layers[0].acoustic.vs" },
    { "AcousticBelowElastic",
      R"yaml(
layers:
  - elastic:  { rho: 2800, vp: 6000, vs: 3500, thickness: 35000 }
  - acoustic: { rho: 1025, vp: 1500, thickness: 2000 }
  - elastic:  { rho: 3300, vp: 8100, vs: 4500, thickness: 0 }
)yaml",
      "layers[1]" },
    // A zero-thickness layer above the half-space fails LayeredModel's check.
    { "InvalidLayerValues",
      R"yaml(
layers:
  - elastic: { rho: 2800, vp: 6000, vs: 3500, thickness: 0 }
  - elastic: { rho: 3300, vp: 8100, vs: 4500, thickness: 0 }
)yaml",
      "fk config: layers:" },
    { "UnknownTopLevelKey", "window-steps: 7\n", "window-steps" },
    { "UnknownIncidenceKey",
      config_text(
          "{ type: P, azimuth: 40, take-off: 20, origin: [0, 0, 0], foo: 1 }"),
      "incidence.foo" },
    { "UnknownTimeWindowKey",
      config_text(valid_incidence,
                  "{ frequency-max: 1.0, length: 128.0, window-steps: 7 }"),
      "time-window.window-steps" },
    { "MissingType",
      config_text("{ azimuth: 40, take-off: 20, origin: [0, 0, 0] }"),
      "incidence.type" },
    { "MissingAzimuth",
      config_text("{ type: P, take-off: 20, origin: [0, 0, 0] }"),
      "back-azimuth or azimuth" },
    { "MissingTakeOff",
      config_text("{ type: P, azimuth: 40, origin: [0, 0, 0] }"),
      "incidence.take-off" },
    { "MissingOrigin", config_text("{ type: P, azimuth: 40, take-off: 20 }"),
      "incidence.origin" },
    { "MissingIncidence", config_text(""), "incidence" },
    { "MissingTimeWindow", config_text(valid_incidence, ""), "time-window" },
    { "MissingLength", config_text(valid_incidence, "{ frequency-max: 1.0 }"),
      "time-window.length" },
    { "InvalidType",
      config_text("{ type: X, azimuth: 40, take-off: 20, origin: [0, 0, 0] }"),
      "incidence.type" },
    { "InvalidTakeOff",
      config_text("{ type: P, azimuth: 40, take-off: abc, origin: [0, 0, 0] }"),
      "incidence.take-off" },
    { "ShortOrigin",
      config_text("{ type: P, azimuth: 40, take-off: 20, origin: [0, 0] }"),
      "incidence.origin" },
    { "ZeroMaterialTolerance", config_text() + "material-tolerance: 0\n",
      "material-tolerance" },
    { "InvalidReproduceFlag",
      config_text() + "reproduce-specfem3d-reference: maybe\n",
      "reproduce-specfem3d-reference" },
    { "ZeroHalfDuration",
      config_text("{ type: P, azimuth: 40, take-off: 20, origin: [0, 0, 0], "
                  "half-duration: 0 }"),
      "incidence.half-duration" },
    // erfc(sqrt(2) pi 1 Hz 0.1 s) = 53% of the energy above the band.
    { "HalfDurationTooShortForTheBand",
      config_text("{ type: P, azimuth: 40, take-off: 20, origin: [0, 0, 0], "
                  "half-duration: 0.1 }"),
      "incidence.half-duration" },
    { "NoBandLimit", config_text(valid_incidence, "{ length: 128.0 }"),
      "no band limit" },
    { "MissingModelFile", "model-file: /no/such/file\n",
      "fk config: model-file:" },
  };
}

// Names the case in test listings, rather than its bytes.
void PrintTo(const RejectionCase &rejection, std::ostream *stream) {
  *stream << rejection.name;
}

class FkPluginConfigRejects : public ::testing::TestWithParam<RejectionCase> {};

// Parsing throws, and the message contains the reason.
TEST_P(FkPluginConfigRejects, Document) {
  try {
    fk_injection::parse_plugin_config(GetParam().document);
    FAIL() << "expected a rejection mentioning '" << GetParam().reason << "'";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(GetParam().reason),
              std::string::npos)
        << error.what();
  }
}

INSTANTIATE_TEST_SUITE_P(
    Documents, FkPluginConfigRejects, ::testing::ValuesIn(rejection_cases()),
    [](const ::testing::TestParamInfo<RejectionCase> &info) {
      return info.param.name;
    });

} // namespace fk_injection_test

TEST(FkPluginConfig, InlineLayersParse) {
  const auto config =
      fk_injection::parse_plugin_config(fk_injection_test::inline_config);

  EXPECT_EQ(config.model.number_of_fluid_layers(), 1);
  EXPECT_EQ(config.model.number_of_elastic_layers(), 2);
  EXPECT_EQ(config.mesh_frame_wave.type, fk_injection::incident_wave_type::p);

  // back-azimuth 30 => phi = -30 - 90 = -120 degrees
  EXPECT_NEAR(config.mesh_frame_wave.azimuth_phi,
              -120.0 * fk_injection_test::degree, 1e-5);
  EXPECT_NEAR(config.mesh_frame_wave.take_off_theta,
              25.0 * fk_injection_test::degree, 1e-5);
  EXPECT_NEAR(config.mesh_frame_wave.amplitude, 2.0, 1e-6);

  const auto window = config.time_window(0.05f, 256);
  EXPECT_NEAR(window.dt, 0.05, 1e-6);
  EXPECT_EQ(window.nstep, 256);
  EXPECT_NEAR(window.frequency_max, 1.0, 1e-6);
  EXPECT_NEAR(window.frequency_sampling, 10.0, 1e-6);
  EXPECT_NEAR(window.time_window_length, 128.0, 1e-6);

  EXPECT_NEAR(config.material_tolerance, 1e-3, 1e-12);
  EXPECT_EQ(config.convention, fk_injection::ReferenceConvention::exact);
}

TEST(FkPluginConfig, AzimuthConvention) {
  const auto config =
      fk_injection::parse_plugin_config(fk_injection_test::config_text());
  // azimuth 40 => phi = 90 - 40 = 50 degrees
  EXPECT_NEAR(config.mesh_frame_wave.azimuth_phi,
              50.0 * fk_injection_test::degree, 1e-5);
}

TEST(FkPluginConfig, HalfDurationDefaultsFromFrequencyMax) {
  const auto config =
      fk_injection::parse_plugin_config(fk_injection_test::config_text(
          fk_injection_test::valid_incidence,
          "{ frequency-max: 2.0, frequency-sampling: 10.0, length: 128.0 }"));
  EXPECT_NEAR(config.mesh_frame_wave.gaussian_half_duration, 0.5, 1e-5);
}

TEST(FkPluginConfig, SamplingDefaultsFromHalfDuration) {
  const auto config =
      fk_injection::parse_plugin_config(fk_injection_test::config_text(
          "{ type: P, azimuth: 40, take-off: 20, origin: [0, 0, 0], "
          "half-duration: 2.0 }",
          "{ length: 128.0 }"));
  EXPECT_NEAR(config.time_window(0.05f, 256).frequency_sampling, 5.0, 1e-5);
  EXPECT_NEAR(config.frequency_max, 0.5, 1e-5);
}

// erfc(sqrt(2) pi 1 Hz 0.2 s) = 21% of the energy above the band.
TEST(FkPluginConfig, HalfDurationInsideTheBandIsAccepted) {
  const auto config =
      fk_injection::parse_plugin_config(fk_injection_test::config_text(
          "{ type: P, azimuth: 40, take-off: 20, origin: [0, 0, 0], "
          "half-duration: 0.2 }"));
  EXPECT_NEAR(config.mesh_frame_wave.gaussian_half_duration, 0.2, 1e-6);
}

TEST(FkPluginConfig, ReproduceSpecfem3dSelectsTheConvention) {
  const auto config = fk_injection::parse_plugin_config(
      std::string(fk_injection_test::inline_config) +
      "reproduce-specfem3d-reference: true\n");
  EXPECT_EQ(config.convention, fk_injection::ReferenceConvention::specfem3d);
}

// By default the wavefront crosses the origin at the first step, whatever t0
// is.
TEST(FkPluginConfig, DefaultOriginTimeIsTheFirstStep) {
  const auto config =
      fk_injection::parse_plugin_config(fk_injection_test::inline_config);
  EXPECT_FALSE(config.origin_time.has_value());
  EXPECT_NEAR(config.incident_wave(-12.0, 0.0).origin_time, 0.0, 1e-3);
  EXPECT_NEAR(config.incident_wave(0.0, 0.0).origin_time, 0.0, 1e-3);
}

// The FK time at step 0 is t0 - t_origin.
TEST(FkPluginConfig, OriginTimeIsSimulationTime) {
  const auto config =
      fk_injection::parse_plugin_config(fk_injection_test::config_text(
          "{ type: P, azimuth: 40, take-off: 20, origin: [0, 0, 0], "
          "origin-time: 5 }"));
  EXPECT_NEAR(config.incident_wave(-12.0, 0.0).origin_time, -17.0, 1e-3);
  EXPECT_NEAR(config.incident_wave(5.0, 0.0).origin_time, 0.0, 1e-3);
}

// The FK frame has the model top at z = 0.
TEST(FkPluginConfig, OriginIsShiftedToTheModelTop) {
  const auto config =
      fk_injection::parse_plugin_config(fk_injection_test::inline_config);
  const auto wave = config.incident_wave(0.0, 1000.0);
  EXPECT_NEAR(wave.origin_x, 1.0, 1e-3);
  EXPECT_NEAR(wave.origin_y, 2.0, 1e-3);
  EXPECT_NEAR(wave.origin_z, -997.0, 1e-3);
}

TEST(FkPluginConfig, ModelFileProvidesEverythingAndYamlOverrides) {
  const fk_injection_test::ModelFile file("fk_plugin_config_model.txt",
                                          fk_injection_test::model_file);

  const auto from_file =
      fk_injection::parse_plugin_config("model-file: " + file.path() + "\n");
  EXPECT_EQ(from_file.model.number_of_elastic_layers(), 2);
  EXPECT_NEAR(from_file.frequency_max, 0.2, 1e-6);
  EXPECT_NEAR(from_file.time_window_length, 180.0, 1e-4);
  EXPECT_NEAR(from_file.mesh_frame_wave.gaussian_half_duration, 5.0, 1e-5);

  const auto overridden =
      fk_injection::parse_plugin_config("model-file: " + file.path() + R"yaml(
incidence: { origin: [0, 0, -223944.719], origin-time: 1.5 }
time-window: { frequency-max: 0.1, length: 90.0 }
)yaml");
  // Keys the override leaves out keep the file's values.
  EXPECT_NEAR(overridden.mesh_frame_wave.take_off_theta,
              20.0 * fk_injection_test::degree, 1e-5);
  EXPECT_NEAR(overridden.mesh_frame_wave.origin_z, -223944.719, 1e-1);
  EXPECT_NEAR(*overridden.origin_time, 1.5, 1e-12);
  EXPECT_NEAR(overridden.frequency_max, 0.1, 1e-6);
  EXPECT_NEAR(overridden.time_window_length, 90.0, 1e-4);
  // The file's half-duration wins over the one derived from frequency-max.
  EXPECT_NEAR(overridden.mesh_frame_wave.gaussian_half_duration, 5.0, 1e-5);
}

TEST(FkPluginConfig, ModelFileWindowOverrideKeepsTheFileLength) {
  const fk_injection_test::ModelFile file("fk_plugin_config_window.txt",
                                          fk_injection_test::model_file);
  const auto config = fk_injection::parse_plugin_config(
      "model-file: " + file.path() + "\ntime-window: { frequency-max: 0.1 }\n");
  EXPECT_NEAR(config.frequency_max, 0.1, 1e-6);
  EXPECT_NEAR(config.time_window_length, 180.0, 1e-4);
}

// A silent default for any of these would run a different wave than the file
// asks for.
TEST(FkPluginConfig, RejectsInvalidModelFiles) {
  const std::string layers = "NLAYER 1\nLAYER 1 2700. 3000. 1732. 0.\n";
  const std::string origin = "ORIGIN_WAVEFRONT 0. 0. -1000.\n";
  const std::vector<fk_injection_test::RejectionCase> cases = {
    { "NoOrigin", layers, "ORIGIN_WAVEFRONT" },
    { "BadWaveType", layers + origin + "INCIDENT_WAVE s\n", "INCIDENT_WAVE" },
    { "UnreadableAmplitude", layers + origin + "AMPLITUDE abc\n", "AMPLITUDE" },
    { "UnreadableOrigin", layers + "ORIGIN_WAVEFRONT 0. 0.\n",
      "ORIGIN_WAVEFRONT" },
    { "UnknownKeyword", layers + origin + "AMPLITUD 2.\n", "AMPLITUD" },
  };
  for (const auto &rejection : cases) {
    const fk_injection_test::ModelFile file("fk_plugin_config_invalid.txt",
                                            rejection.document);
    try {
      fk_injection::parse_plugin_config("model-file: " + file.path() + "\n");
      FAIL() << rejection.name << ": expected a rejection";
    } catch (const std::runtime_error &error) {
      EXPECT_NE(std::string(error.what()).find(rejection.reason),
                std::string::npos)
          << rejection.name << ": " << error.what();
    }
  }
}

// Comments may be indented, and YAML can supply the origin the file leaves out.
TEST(FkPluginConfig, ModelFileOriginMayComeFromYaml) {
  const fk_injection_test::ModelFile file(
      "fk_plugin_config_yaml_origin.txt",
      "  # indented comment\nNLAYER 1\nLAYER 1 2700. 3000. 1732. 0.\n"
      "TAKE_OFF 20.\nFREQUENCY_MAX 0.2\n");
  const auto config = fk_injection::parse_plugin_config(
      "model-file: " + file.path() +
      "\nincidence: { origin: [1, 2, -3000] }\n");
  EXPECT_NEAR(config.mesh_frame_wave.origin_z, -3000.0, 1e-3);
}

// ORIGIN_TIME is the FK time at step 0, not the time the front crosses origin.
TEST(FkPluginConfig, RejectsNonZeroModelFileOriginTime) {
  const fk_injection_test::ModelFile file(
      "fk_plugin_config_origin_time.txt",
      std::string(fk_injection_test::model_file) + "ORIGIN_TIME 10\n");
  try {
    fk_injection::parse_plugin_config("model-file: " + file.path() + "\n");
    FAIL() << "expected a rejection mentioning incidence.origin-time";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("incidence.origin-time"),
              std::string::npos)
        << error.what();
  }
}

// The example the README points to stays valid.
TEST(FkPluginConfig, ExampleConfigurationLoads) {
  const auto config =
      fk_injection::load_plugin_config(FK_INJECTION_EXAMPLE_CONFIG);
  EXPECT_EQ(config.model.number_of_elastic_layers(), 2);
  EXPECT_NEAR(config.time_window_length, 180.0, 1e-4);
}

// The values of the FKInjectionElastic integration fixture.
TEST(FkPluginConfig, ElasticFixtureValues) {
  const auto config = fk_injection::parse_plugin_config(R"yaml(
layers:
  - elastic: { rho: 2500.0, vp: 2500.0, vs: 1250.000, thickness: 30000.0 }
  - elastic: { rho: 2700.0, vp: 3000.0, vs: 1732.051, thickness: 0.0 }
incidence:
  type: P
  back-azimuth: 0.0
  take-off: 20.0
  origin: [0.0, 0.0, -223944.719]
time-window: { frequency-max: 0.2, frequency-sampling: 10.0, length: 180.0 }
)yaml");
  ASSERT_EQ(config.model.number_of_fluid_layers(), 0);
  ASSERT_EQ(config.model.number_of_elastic_layers(), 2);
  EXPECT_NEAR(config.model.elastic_layers()[0].density, 2500.0, 1e-3);
  EXPECT_NEAR(config.model.elastic_layers()[0].p_velocity, 2500.0, 1e-3);
  EXPECT_NEAR(config.model.elastic_layers()[0].s_velocity, 1250.0, 1e-3);
  EXPECT_NEAR(config.model.elastic_layers()[0].thickness, 30000.0, 1e-3);
  EXPECT_NEAR(config.model.elastic_layers()[1].s_velocity, 1732.051, 1e-3);

  const auto wave = config.incident_wave(-12.0, 0.0);
  EXPECT_NEAR(wave.azimuth_phi, -90.0 * fk_injection_test::degree, 1e-5);
  EXPECT_NEAR(wave.amplitude, 1.0, 1e-3);
  EXPECT_NEAR(wave.origin_z, -223944.719, 1e-1);
  EXPECT_NEAR(wave.origin_time, 0.0, 1e-3);
  EXPECT_NEAR(wave.gaussian_half_duration, 5.0, 1e-5);

  const auto window = config.time_window(0.3f, 600);
  EXPECT_NEAR(window.frequency_max, 0.2, 1e-6);
  EXPECT_NEAR(window.frequency_sampling, 10.0, 1e-6);
  EXPECT_NEAR(window.time_window_length, 180.0, 1e-4);
}

// The values of the FKInjectionAcousticElastic integration fixture.
TEST(FkPluginConfig, AcousticFixtureValues) {
  const auto config = fk_injection::parse_plugin_config(R"yaml(
layers:
  - acoustic: { rho: 1000.0, vp: 1500.0, thickness: 30000.0 }
  - elastic: { rho: 2700.0, vp: 3000.0, vs: 1732.051, thickness: 0.0 }
incidence:
  type: P
  back-azimuth: 30.0
  take-off: 20.0
  origin: [0.0, 0.0, -223944.719]
time-window: { frequency-max: 0.2, frequency-sampling: 10.0, length: 180.0 }
)yaml");
  ASSERT_EQ(config.model.number_of_fluid_layers(), 1);
  ASSERT_EQ(config.model.number_of_elastic_layers(), 1);
  EXPECT_NEAR(config.model.fluid_layers()[0].density, 1000.0, 1e-3);
  EXPECT_NEAR(config.model.fluid_layers()[0].p_velocity, 1500.0, 1e-3);
  EXPECT_NEAR(config.model.fluid_layers()[0].thickness, 30000.0, 1e-3);
  EXPECT_NEAR(config.mesh_frame_wave.azimuth_phi,
              -120.0 * fk_injection_test::degree, 1e-5);
}
