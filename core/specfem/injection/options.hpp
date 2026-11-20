#pragma once

#include <optional>
#include <string>

namespace specfem {
namespace injection {

/**
 * @brief Injection plugin selected for a run.
 */
struct PluginOptions {
  std::string plugin; ///< Plugin path, or "static:<name>"
  std::string config; ///< Plugin configuration file, passed to the plugin
};

/**
 * @brief Resolve the plugin options from the command line.
 *
 * A configuration file is required whenever a plugin is selected, and is an
 * error without one.
 *
 * @param cli_plugin Value of @c --injection-plugin, if given
 * @param cli_config Value of @c --injection-config, if given
 * @return The selected plugin, or @c std::nullopt when none is selected
 * @throws std::runtime_error if the options are inconsistent
 */
[[nodiscard]] std::optional<PluginOptions>
resolve_plugin_options(const std::optional<std::string> &cli_plugin,
                       const std::optional<std::string> &cli_config);

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
