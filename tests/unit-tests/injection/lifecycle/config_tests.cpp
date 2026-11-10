// Tests for the parameters.injection section and the injection-method
// registry.

#include "recording_method.hpp"
#include "specfem/injection/method/registry.hpp"
#include "specfem/runtime_configuration/injection.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

namespace specfem::injection_lifecycle_test {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

std::string config_error(const std::string &yaml) {
  try {
    specfem::runtime_configuration::Injection config(YAML::Load(yaml));
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "<no error>";
}

TEST(InjectionConfig, MethodOnlyUsesDefaults) {
  const specfem::runtime_configuration::Injection config(
      YAML::Load("method: stacey"));
  EXPECT_EQ(config.method(), "stacey");
  EXPECT_FALSE(config.window_steps().has_value());
  EXPECT_FALSE(config.memory_budget_mib().has_value());
  EXPECT_FALSE(config.validate().has_value());
}

TEST(InjectionConfig, ReadsAllKeys) {
  const specfem::runtime_configuration::Injection config(
      YAML::Load("{method: stacey, window-steps: 64, memory-budget: 128, "
                 "validate: true}"));
  EXPECT_EQ(config.window_steps(), 64);
  EXPECT_EQ(config.memory_budget_mib(), 128);
  EXPECT_EQ(config.validate(), true);
}

TEST(InjectionConfig, RejectsBadSections) {
  EXPECT_THAT(config_error("{window-steps: 4}"),
              HasSubstr("parameters.injection.method is required"));
  EXPECT_THAT(config_error("{method: stacey, window_steps: 4}"),
              HasSubstr("Unknown key parameters.injection.window_steps"));
  EXPECT_THAT(config_error("{method: stacey, frequency: 0.2}"),
              HasSubstr("belong in the plugin's configuration file"));
  EXPECT_THAT(config_error("{method: stacey, window-steps: 0}"),
              HasSubstr("window-steps must be at least 1"));
  EXPECT_THAT(config_error("{method: stacey, memory-budget: -1}"),
              HasSubstr("memory-budget must be at least 1"));
  EXPECT_THAT(config_error("stacey"),
              HasSubstr("parameters.injection must be a map"));
}

TEST(InjectionRegistry, MakesRegisteredMethods) {
  specfem::injection::MethodRegistry registry;
  registry.add<RecordingMethod>();
  EXPECT_TRUE(registry.contains("recording"));
  EXPECT_FALSE(registry.contains("stacey"));
  EXPECT_THAT(registry.names(), ElementsAre("recording"));
  EXPECT_EQ(registry.make("recording")->name(), "recording");
  EXPECT_THROW(registry.add<RecordingMethod>(), std::runtime_error);
}

TEST(InjectionRegistry, UnknownMethodListsRegistered) {
  specfem::injection::MethodRegistry registry;
  EXPECT_THAT(registry.unknown_method_message("stacey"),
              HasSubstr("Unknown injection method 'stacey'. Registered "
                        "methods: (none)."));
  registry.add<RecordingMethod>();
  try {
    [[maybe_unused]] const auto method = registry.make("stacey");
    FAIL() << "an unknown method should throw";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(), HasSubstr("Registered methods: recording."));
  }
}

TEST(InjectionRegistry, DefaultRegistryHasStacey) {
  const auto &registry = specfem::injection::default_method_registry();
  EXPECT_THAT(registry.names(), ElementsAre("stacey"));
  EXPECT_EQ(registry.make("stacey")->name(), "stacey");
}

} // namespace specfem::injection_lifecycle_test
