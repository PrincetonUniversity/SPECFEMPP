// Tests for the injection option rules: command line versus environment, and
// consistency with the parameter file's injection section.

#include "specfem/injection.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace specfem::injection_loader_test {

using ::testing::HasSubstr;

// Message of the std::runtime_error thrown by resolve_plugin_options.
std::string resolve_error(const std::optional<std::string> &plugin,
                          const std::optional<std::string> &config,
                          const char *environment) {
  try {
    [[maybe_unused]] const auto options =
        specfem::injection::resolve_plugin_options(plugin, config, environment);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "<no error>";
}

TEST(InjectionOptions, NothingSelected) {
  EXPECT_FALSE(specfem::injection::resolve_plugin_options(std::nullopt,
                                                          std::nullopt, nullptr)
                   .has_value());
  // An empty environment variable counts as unset.
  EXPECT_FALSE(
      specfem::injection::resolve_plugin_options(std::nullopt, std::nullopt, "")
          .has_value());
}

TEST(InjectionOptions, FlagWinsOverEnvironment) {
  const auto options = specfem::injection::resolve_plugin_options(
      "/flag/plugin.so", "plugin.yaml", "/env/plugin.so");
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->plugin, "/flag/plugin.so");
  EXPECT_EQ(options->config, "plugin.yaml");
}

TEST(InjectionOptions, EnvironmentFallback) {
  const auto options = specfem::injection::resolve_plugin_options(
      std::nullopt, "plugin.yaml", "/env/plugin.so");
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->plugin, "/env/plugin.so");
}

TEST(InjectionOptions, ConfigRequiredWithPlugin) {
  EXPECT_THAT(resolve_error("/flag/plugin.so", std::nullopt, nullptr),
              HasSubstr("without a plugin configuration"));
  EXPECT_THAT(resolve_error(std::nullopt, std::nullopt, "/env/plugin.so"),
              HasSubstr("without a plugin configuration"));
}

TEST(InjectionOptions, ConfigWithoutPlugin) {
  EXPECT_THAT(resolve_error(std::nullopt, "plugin.yaml", nullptr),
              HasSubstr("--injection-config was given without an injection "
                        "plugin"));
}

TEST(InjectionOptions, Consistency) {
  const specfem::injection::PluginOptions options{ "/p.so", "p.yaml" };
  EXPECT_NO_THROW(
      specfem::injection::check_injection_consistency(false, std::nullopt));
  EXPECT_NO_THROW(
      specfem::injection::check_injection_consistency(true, options));

  try {
    specfem::injection::check_injection_consistency(false, options);
    FAIL() << "a plugin without an injection section should throw";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(), HasSubstr("has no 'injection' section"));
  }
  try {
    specfem::injection::check_injection_consistency(true, std::nullopt);
    FAIL() << "an injection section without a plugin should throw";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(), HasSubstr("no injection plugin was given"));
  }
}

} // namespace specfem::injection_loader_test
