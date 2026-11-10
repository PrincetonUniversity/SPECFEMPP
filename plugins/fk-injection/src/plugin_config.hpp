#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/solver.hpp"
#include "fk_injection/time_window.hpp"

#include <optional>
#include <string>

namespace fk_injection {

/** @brief Contents of the plugin's configuration file. */
struct PluginConfig {
  LayeredModel model; ///< Layered velocity model
  /// Wave as configured, origin in the mesh frame; @c solve takes
  /// @c incident_wave instead
  IncidentWave mesh_frame_wave = {};
  /// Simulation time the wavefront crosses the origin; empty for the first step
  std::optional<double> origin_time = std::nullopt;
  type_real frequency_max = 0;      ///< Maximum frequency of interest in Hz
  type_real frequency_sampling = 0; ///< FK storage sampling frequency in Hz
  type_real time_window_length = 0; ///< FK time-window length in s
  double material_tolerance =
      1e-3; ///< Relative tolerance of the material check
  /// z of the top of the layered model in the mesh frame; empty for the top of
  /// the mesh
  std::optional<double> model_top = std::nullopt;
  ReferenceConvention convention = ReferenceConvention::exact; ///< See @c solve

  /**
   * @brief The time window of a simulation.
   * @param dt Simulation time step in s
   * @param nstep Number of simulation time steps
   * @return Time window
   */
  TimeWindow time_window(type_real dt, int nstep) const;

  /**
   * @brief The incident wave in the frame @c solve works in.
   *
   * @c IncidentWave::origin_time is the FK time at step 0, so it is
   * \f$ t_0 - t_\text{origin} \f$. The FK frame has the model top at z = 0, so
   * the origin's z is shifted by @p zmax.
   *
   * @param t0 Simulation time of step 0 in s
   * @param zmax Top of the mesh in m, used unless @c model_top is set
   * @return Incident wave
   */
  IncidentWave incident_wave(double t0, double zmax) const;

  /**
   * @brief z of the top of the layered model in the mesh frame.
   *
   * Depth in the model is this z minus the point's z. A mesh with topography
   * above the model's top, such as islands in an ocean, sets @c model_top.
   *
   * @param zmax Top of the mesh in m
   * @return @c model_top if set, else @p zmax
   */
  double top(double zmax) const { return model_top.value_or(zmax); }
};

/**
 * @brief Read an FK configuration file.
 *
 * @code{.yaml}
 * layers:                       # or model-file: path, exactly one of the two
 *   - acoustic: {rho: 1000, vp: 1500, thickness: 30000}
 *   - elastic:  {rho: 2700, vp: 3000, vs: 1732.051, thickness: 0}
 * incidence:
 *   type: P                     # P | SV
 *   back-azimuth: 30            # or azimuth
 *   take-off: 20                # degrees from vertical
 *   origin: [0, 0, -223944.719] # mesh frame, z up
 *   origin-time: 5              # optional, default: the first step
 *   amplitude: 1                # optional
 *   half-duration: 5            # optional, default: 1 / frequency-max
 * time-window:
 *   frequency-max: 0.2          # optional, default: 1 / half-duration
 *   frequency-sampling: 10      # optional, default: 10 / half-duration
 *   length: 180
 * material-tolerance: 1.0e-3    # optional, relative
 * model-top: 0                 # optional, default: the top of the mesh
 * reproduce-specfem3d-reference: false  # optional
 * @endcode
 *
 * A half-duration that leaves more than 40% of the Gaussian's energy above
 * @c frequency-max is rejected, which includes a half-duration of 0.
 *
 * With @c model-file, @c incidence and @c time-window are optional and override
 * the values read from the file. A non-zero @c ORIGIN_TIME in the file is
 * rejected: it means the FK time at step 0, not @c incidence.origin-time.
 *
 * @param path Path of the YAML file
 * @return Configuration
 * @throws std::runtime_error naming the offending key on a missing, unknown or
 * invalid entry
 */
PluginConfig load_plugin_config(const std::string &path);

/**
 * @brief Parse an FK configuration from YAML text.
 * @param text YAML document
 * @return Configuration
 * @throws std::runtime_error as @c load_plugin_config
 */
PluginConfig parse_plugin_config(const std::string &text);

} // namespace fk_injection
