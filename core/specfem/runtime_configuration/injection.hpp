#pragma once

#include "yaml-cpp/yaml.h"
#include <optional>
#include <string>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Wavefield-injection section of the parameter file
 * (@c parameters.injection).
 *
 * Describes how SPECFEM++ applies and buffers the incident field: the
 * injection method and the provider session's windowing. Everything about the
 * incident field itself belongs in the plugin's own configuration file, so
 * unknown keys are rejected.
 *
 * @code{.yaml}
 * injection:
 *   method: stacey      # required
 *   window-steps: 64    # optional
 *   memory-budget: 256  # optional, MiB
 *   validate: false     # optional
 * @endcode
 */
class Injection {
public:
  /**
   * @brief Read the section.
   * @param node The @c injection node
   * @throws std::runtime_error if @c method is missing, a key is unknown, or
   * a value is not positive
   */
  explicit Injection(const YAML::Node &node);

  /**
   * @brief Name of the injection method.
   * @return Method name, as registered
   */
  [[nodiscard]] const std::string &method() const { return method_; }

  /**
   * @brief Steps per plugin fill, if set.
   * @return Window size, or nullopt to fit the memory budget
   */
  [[nodiscard]] std::optional<int> window_steps() const {
    return window_steps_;
  }

  /**
   * @brief Memory for the staging and device buffers, if set.
   * @return Budget in MiB, or nullopt for the default
   */
  [[nodiscard]] std::optional<int> memory_budget_mib() const {
    return memory_budget_mib_;
  }

  /**
   * @brief Whether to reject non-finite plugin values, if set.
   * @return Validation flag, or nullopt for the build default
   */
  [[nodiscard]] std::optional<bool> validate() const { return validate_; }

private:
  std::string method_;                   ///< Injection method name
  std::optional<int> window_steps_;      ///< Steps per window
  std::optional<int> memory_budget_mib_; ///< Buffer budget in MiB
  std::optional<bool> validate_;         ///< Reject non-finite values
};

} // namespace runtime_configuration
} // namespace specfem
