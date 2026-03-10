#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"

namespace specfem::utilities {

/**
 * @brief Min/max attenuation period band
 */
struct FrequencyBand {
  type_real min_omega; ///< Minimum attenuation angular frequency (rad/s)
  type_real max_omega; ///< Maximum attenuation angular frequency (rad/s)

  FrequencyBand() = default;

  /// Construct from period bounds (s).
  static FrequencyBand from_period(const type_real min_period,
                                   const type_real max_period) {
    return { static_cast<type_real>(2.0 * specfem::constants::pi) / max_period,
             static_cast<type_real>(2.0 * specfem::constants::pi) /
                 min_period };
  }

  /// Construct from frequency bounds (Hz).
  static FrequencyBand from_frequency(const type_real min_frequency,
                                      const type_real max_frequency) {
    return {
      static_cast<type_real>(2.0 * specfem::constants::pi) * min_frequency,
      static_cast<type_real>(2.0 * specfem::constants::pi) * max_frequency
    };
  }

  /// Construct from angular-frequency bounds (rad/s).
  static FrequencyBand from_omega(const type_real min_omega,
                                  const type_real max_omega) {
    return { min_omega, max_omega };
  }

  /// Minimum period (s).
  type_real min_period() const {
    return static_cast<type_real>(2.0 * specfem::constants::pi) / max_omega;
  }
  /// Maximum period (s).
  type_real max_period() const {
    return static_cast<type_real>(2.0 * specfem::constants::pi) / min_omega;
  }
  /// Minimum frequency (Hz).
  type_real min_frequency() const {
    return min_omega / static_cast<type_real>(2.0 * specfem::constants::pi);
  }
  /// Maximum frequency (Hz).
  type_real max_frequency() const {
    return max_omega / static_cast<type_real>(2.0 * specfem::constants::pi);
  }
};

} // namespace specfem::utilities
