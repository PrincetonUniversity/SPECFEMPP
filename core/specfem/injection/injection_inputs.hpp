#pragma once

#include "specfem/injection/fk/field_derivative.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/setup.hpp"
#include <string>

namespace specfem {
namespace injection {

/**
 * @brief Domain inputs needed to build an injection provider.
 *
 * Carries everything the @c injection: YAML section resolves to, independent of
 * @c specfem::runtime_configuration. The assembly builds its boundary injection
 * provider from this POD, so the assembly library depends only on
 * @c specfem::injection (never on runtime_configuration).
 *
 * A default-constructed value is @c disabled, so passing one to a constructor
 * that threads injection inputs is a safe no-op.
 */
struct injection_inputs {
  bool enabled = false;      ///< Master switch (default: injection off)
  std::string method = "fk"; ///< Provider method (lower-cased)
  specfem::injection::fk::LayeredModel model{}; ///< 1-D layered velocity model
  specfem::injection::fk::IncidentWave wave{};  ///< Incident plane wave
  specfem::injection::fk::field_derivative derivative =
      specfem::injection::fk::field_derivative::displacement; ///< Vector
                                                              ///< quantity
  type_real frequency_max = 0;      ///< FK storage max frequency in Hz
  type_real frequency_sampling = 0; ///< FK storage sampling frequency in Hz
  type_real time_window_length = 0; ///< FK time-window length in s
  int window_steps = 0;             ///< Frame-buffer window (0 = automatic)
  bool compute_traction = true;     ///< Store traction (the Stacey method needs
                                    ///< it)
  double consistency_tolerance =
      1e-3; ///< Relative tolerance for the 1-D-model vs. mesh moduli check
};

} // namespace injection
} // namespace specfem
