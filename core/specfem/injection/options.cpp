#include "specfem/injection/options.hpp"
#include <stdexcept>

std::optional<specfem::injection::PluginOptions>
specfem::injection::resolve_plugin_options(
    const std::optional<std::string> &cli_plugin,
    const std::optional<std::string> &cli_config,
    const char *environment_plugin) {
  std::optional<std::string> plugin = cli_plugin;
  if (!plugin && environment_plugin != nullptr && *environment_plugin != '\0')
    plugin = std::string(environment_plugin);

  if (!plugin) {
    if (cli_config)
      throw std::runtime_error(
          "--injection-config was given without an injection plugin. Pass "
          "--injection-plugin <absolute path> or set " +
          std::string(specfem::injection::plugin_environment_variable) + ".");
    return std::nullopt;
  }

  if (plugin->empty())
    throw std::runtime_error("--injection-plugin must not be empty.");

  if (!cli_config || cli_config->empty())
    throw std::runtime_error(
        "Injection plugin '" + *plugin +
        "' was given without a plugin configuration. Pass "
        "--injection-config <path> with the plugin's configuration file.");

  return specfem::injection::PluginOptions{ *plugin, *cli_config };
}

void specfem::injection::check_injection_consistency(
    const bool has_injection_section,
    const std::optional<specfem::injection::PluginOptions> &options) {
  if (options && !has_injection_section)
    throw std::runtime_error(
        "Injection plugin '" + options->plugin +
        "' was given, but the parameter file has no 'injection' section. Add "
        "parameters.injection to select an injection method, or drop the "
        "plugin.");
  if (!options && has_injection_section)
    throw std::runtime_error(
        "The parameter file has an 'injection' section, but no injection "
        "plugin was given. Pass --injection-plugin <absolute path> and "
        "--injection-config <path>, or set " +
        std::string(specfem::injection::plugin_environment_variable) + ".");
}
