#pragma once

#include "plane_wave/plane_wave.hpp"
#include <string>

namespace plane_wave {

/** @brief Contents of the plugin's configuration file. */
struct Config {
  Material material;            ///< Material of the whole space
  WaveType type = WaveType::p;  ///< P or SV
  double take_off = 0;          ///< Take-off angle from +z, degrees
  double azimuth = 0;           ///< Azimuth from +x toward +y, degrees
  Vector origin{};              ///< Point the wavefront crosses at origin_time
  double origin_time = 0;       ///< Time the wavefront crosses origin
  double amplitude = 1;         ///< Displacement amplitude
  SourceTimeFunction waveform = ///< Displacement waveform
      SourceTimeFunction::ricker(1);
  double material_tolerance = 0; ///< Relative tolerance of the material check

  /** @brief The plane wave described. @return Plane wave */
  PlaneWave plane_wave() const {
    return { material, type,        take_off,  azimuth,
             origin,   origin_time, amplitude, waveform };
  }
};

/**
 * @brief Read a plane-wave configuration file.
 *
 * @code{.yaml}
 * material: {rho: 2700, vp: 6000, vs: 3464}
 * wave:
 *   type: P              # P | SV
 *   take-off: 30         # degrees from +z
 *   azimuth: 45          # degrees from +x toward +y
 *   origin: [0, 0, 0]
 *   origin-time: 0
 *   amplitude: 1.0e-3
 * source-time-function: {type: ricker, f0: 2.0}
 * material-tolerance: 1.0e-3
 * @endcode
 *
 * The source time function is either @c {type: ricker, f0} or
 * @c {type: gaussian, half-duration}.
 *
 * @param path Path of the YAML file
 * @return Configuration
 * @throws std::runtime_error naming the offending key on a missing, unknown
 * or invalid entry
 */
Config load_config(const std::string &path);

/**
 * @brief Parse a plane-wave configuration from YAML text.
 * @param text YAML document
 * @return Configuration
 * @throws std::runtime_error as load_config()
 */
Config parse_config(const std::string &text);

} // namespace plane_wave
