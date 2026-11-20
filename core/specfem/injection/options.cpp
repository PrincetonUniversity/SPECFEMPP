#include "specfem/injection/options.hpp"
#include <stdexcept>

std::optional<specfem::injection::PluginOptions>
specfem::injection::resolve_plugin_options(
    const std::optional<std::string> &cli_plugin,
    const std::optional<std::string> &cli_config) {
  if (!cli_plugin) {
    if (cli_config)
      throw std::runtime_error(
          "--injection-config was given without an injection plugin. Pass "
          "--injection-plugin <path>.");
    return std::nullopt;
  }

  if (cli_plugin->empty())
    throw std::runtime_error("--injection-plugin must not be empty.");

  if (!cli_config || cli_config->empty())
    throw std::runtime_error(
        "Injection plugin '" + *cli_plugin +
        "' was given without a plugin configuration. Pass "
        "--injection-config <path> with the plugin's configuration file.");

  return specfem::injection::PluginOptions{ *cli_plugin, *cli_config };
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
        "plugin was given. Pass --injection-plugin <path> and "
        "--injection-config <path>.");
}
