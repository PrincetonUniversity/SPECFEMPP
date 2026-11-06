// Tests for the injection option rules: the command-line flags, and
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
                          const std::optional<std::string> &config) {
  try {
    [[maybe_unused]] const auto options =
        specfem::injection::resolve_plugin_options(plugin, config);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "<no error>";
}

TEST(InjectionOptions, NothingSelected) {
  EXPECT_FALSE(
      specfem::injection::resolve_plugin_options(std::nullopt, std::nullopt)
          .has_value());
}

TEST(InjectionOptions, PluginAndConfig) {
  const auto options = specfem::injection::resolve_plugin_options(
      "/flag/plugin.so", "plugin.yaml");
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->plugin, "/flag/plugin.so");
  EXPECT_EQ(options->config, "plugin.yaml");
}

TEST(InjectionOptions, EmptyPlugin) {
  EXPECT_THAT(resolve_error("", "plugin.yaml"),
              HasSubstr("--injection-plugin must not be empty"));
}

TEST(InjectionOptions, ConfigRequiredWithPlugin) {
  EXPECT_THAT(resolve_error("/flag/plugin.so", std::nullopt),
              HasSubstr("without a plugin configuration"));
  EXPECT_THAT(resolve_error("/flag/plugin.so", ""),
              HasSubstr("without a plugin configuration"));
}

TEST(InjectionOptions, ConfigWithoutPlugin) {
  EXPECT_THAT(resolve_error(std::nullopt, "plugin.yaml"),
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
