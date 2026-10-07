#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <string>

// Forward declarations — keep Kokkos-heavy headers out of this translation
// unit.
namespace specfem {
namespace injection {
template <specfem::element::dimension_tag DimensionTag>
class injection_provider;
namespace fk {
struct EvalPoints;
} // namespace fk
} // namespace injection
} // namespace specfem

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Runtime configuration for the wavefield-injection subsystem.
 *
 * Parses the @c injection: section of the simulation YAML and exposes
 * typed SPECFEM++ injection objects.  Two input forms are supported:
 *
 * **Inline-layers form** — velocity model specified directly in YAML:
 * @code{.yaml}
 * injection:
 *   enabled: true
 *   method: fk
 *   layers:
 *     - acoustic: { rho: 1000, vp: 1500, thickness: 2000 }
 *     - elastic:  { rho: 2800, vp: 6000, vs: 3500, thickness: 35000 }
 *     - elastic:  { rho: 3300, vp: 8100, vs: 4500, thickness: 0 }
 *   incidence:
 *     type: P
 *     back-azimuth: 30.0
 *     take-off: 25.0
 *     origin: [0, 0, 0]
 *     origin-time: 0.0
 *     amplitude: 1.0
 *     half-duration: 0.0
 *   time-window:
 *     frequency-max: 1.0
 *     frequency-sampling: 10.0
 *     length: 128.0
 * @endcode
 *
 * **File form** — all parameters read from a keyword-format FK model file:
 * @code{.yaml}
 * injection:
 *   enabled: true
 *   method: fk
 *   model-file: path/to/fk_model.dat
 * @endcode
 * When @c model-file is present the @c incidence and @c time-window blocks are
 * optional; if supplied they override the values read from the file.
 *
 * Exactly one of @c layers or @c model-file must be present.
 */
class Injection {
public:
  /**
   * @brief Construct from the @c injection: YAML node.
   *
   * Parses @c enabled, @c method, and exactly one of @c layers / @c model-file.
   * For the inline form the @c incidence and @c time-window sub-blocks are also
   * required.  YAML parse/convert exceptions are wrapped into
   * @c std::runtime_error with a prefix message; explicit validation errors
   * propagate as-is.
   *
   * @param node The @c injection: YAML mapping node.
   * @throws std::runtime_error on missing/conflicting keys or invalid values.
   */
  explicit Injection(const YAML::Node &node);

  /**
   * @brief Return whether injection is enabled.
   *
   * Reads the @c enabled: key (default @c true when absent).
   *
   * @return @c true if injection is enabled.
   */
  bool is_enabled() const { return enabled_; }

  /**
   * @brief Return the injection method string (lower-cased).
   *
   * Default is @c "fk" when the @c method: key is absent.
   *
   * @return const reference to the method string.
   */
  const std::string &method() const { return method_; }

  /**
   * @brief Convert the parsed layers to a @c LayeredModel.
   *
   * @return Copy of the internally stored layered velocity model.
   */
  specfem::injection::fk::LayeredModel to_layered_model() const {
    return model_;
  }

  /**
   * @brief Convert the parsed incidence block to an @c IncidentWave.
   *
   * @return Copy of the internally stored incident-wave descriptor.
   */
  specfem::injection::fk::IncidentWave to_incident_wave() const {
    return wave_;
  }

  /**
   * @brief Build a @c TimeWindow using simulation time parameters.
   *
   * The @c dt and @c nstep fields come from the simulation time scheme; the
   * @c frequency_max, @c frequency_sampling, and @c time_window_length fields
   * come from the parsed @c time-window: block (or from the model file).
   *
   * @param dt    Simulation time step in seconds.
   * @param nstep Total number of simulation time steps.
   * @return Populated @c TimeWindow struct.
   */
  specfem::injection::fk::TimeWindow to_time_window(type_real dt,
                                                    int nstep) const;

  /**
   * @brief Factory: map the @c method string to a concrete injection provider.
   *
   * Currently only @c "fk" is recognised; any other method string causes a
   * @c std::runtime_error naming the supported methods.  The provider is
   * constructed but @c initialize() is NOT called.
   *
   * @tparam DimensionTag Spatial dimension of the target simulation
   *                      (@c dim2 or @c dim3).
   * @param points          Device-resident evaluation-point coordinates and
   *                        optional normals / Lamé ratios.
   * @param dt              Simulation time step in seconds.
   * @param nstep           Total number of simulation time steps.
   * @param compute_traction When @c true, traction components are stored in the
   *                         frame buffer (requires non-empty normals/Lamé in
   *                         @p points).
   * @return Shared pointer to the constructed @c injection_provider.
   * @throws std::runtime_error if @c method() is not @c "fk".
   */
  template <specfem::element::dimension_tag DimensionTag>
  std::shared_ptr<specfem::injection::injection_provider<DimensionTag>>
  instantiate(const specfem::injection::fk::EvalPoints &points, type_real dt,
              int nstep, bool compute_traction = true) const;

private:
  bool enabled_ = true;       ///< Whether injection is enabled
  std::string method_ = "fk"; ///< Injection method (lower-cased)
  specfem::injection::fk::LayeredModel model_{}; ///< Layered velocity model
  specfem::injection::fk::IncidentWave wave_{};  ///< Incident wave descriptor
  type_real frequency_max_ = 0;      ///< FK storage max frequency in Hz
  type_real frequency_sampling_ = 0; ///< FK storage sampling frequency in Hz
  type_real time_window_length_ = 0; ///< FK time-window length in s
};

} // namespace runtime_configuration
} // namespace specfem
