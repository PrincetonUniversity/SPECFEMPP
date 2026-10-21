#pragma once

#include <optional>
#include <string>

namespace specfem {
namespace injection {

/**
 * @brief Name of the environment variable used when no plugin is given on the
 * command line.
 */
inline constexpr const char *plugin_environment_variable =
    "SPECFEMPP_INJECTION_PLUGIN";

/**
 * @brief Injection plugin selected for a run.
 */
struct PluginOptions {
  std::string plugin; ///< Absolute plugin path, or "static:<name>"
  std::string config; ///< Plugin configuration file, passed to the plugin
};

/**
 * @brief Resolve the plugin options from the command line and the
 * environment.
 *
 * The command-line plugin takes precedence over @p environment_plugin. A
 * configuration file is required whenever a plugin is selected, and is an
 * error without one.
 *
 * @param cli_plugin Value of @c --injection-plugin, if given
 * @param cli_config Value of @c --injection-config, if given
 * @param environment_plugin Value of @c SPECFEMPP_INJECTION_PLUGIN, or null
 * when unset
 * @return The selected plugin, or @c std::nullopt when none is selected
 * @throws std::runtime_error if the options are inconsistent
 */
std::optional<PluginOptions>
resolve_plugin_options(const std::optional<std::string> &cli_plugin,
                       const std::optional<std::string> &cli_config,
                       const char *environment_plugin);

/**
 * @brief Check that a plugin is given if and only if the main configuration
 * has an @c injection section.
 *
 * @param has_injection_section Whether @c parameters.injection is present
 * @param options The selected plugin, if any
 * @throws std::runtime_error if exactly one of the two is present
 */
void check_injection_consistency(const bool has_injection_section,
                                 const std::optional<PluginOptions> &options);

} // namespace injection
} // namespace specfem
