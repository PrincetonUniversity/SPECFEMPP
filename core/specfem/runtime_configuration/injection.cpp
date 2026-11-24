#include "injection.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>

namespace specfem::runtime_configuration::injection_impl {

/** @brief Keys allowed in the injection section. */
inline constexpr std::array<std::string_view, 4> keys = {
  "method", "window-steps", "memory-budget", "validate"
};

} // namespace specfem::runtime_configuration::injection_impl

specfem::runtime_configuration::Injection::Injection(const YAML::Node &node) {
  namespace impl = specfem::runtime_configuration::injection_impl;
  if (!node.IsMap())
    throw std::runtime_error("parameters.injection must be a map.");

  for (const auto &entry : node) {
    const auto key = entry.first.as<std::string>();
    if (std::find(impl::keys.begin(), impl::keys.end(), key) ==
        impl::keys.end())
      throw std::runtime_error(
          "Unknown key parameters.injection." + key +
          ". Allowed keys: method, window-steps, memory-budget, validate. "
          "Settings of the incident field belong in the plugin's "
          "configuration file.");
  }

  if (!node["method"])
    throw std::runtime_error("parameters.injection.method is required.");
  method_ = node["method"].as<std::string>();

  const auto positive = [&](const std::string &key) {
    const int value = node[key].as<int>();
    if (value < 1)
      throw std::runtime_error("parameters.injection." + key +
                               " must be at least 1, got " +
                               std::to_string(value) + ".");
    return value;
  };
  if (node["window-steps"])
    window_steps_ = positive("window-steps");
  if (node["memory-budget"])
    memory_budget_mib_ = positive("memory-budget");
  if (node["validate"])
    validate_ = node["validate"].as<bool>();
}
