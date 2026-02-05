#pragma once

#include "enumerations/interface.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace specfem::io::impl {

// clang-format off
/**
 * @brief Utility class for generating SEED-compliant seismogram filenames and channel codes.
 *
 * The ChannelGenerator class provides functionality to create standardized seismogram
 * output filenames following SEED (Standard for the Exchange of Earthquake Data) and
 * FDSN (International Federation of Digital Seismograph Networks) naming conventions.
 * It automatically determines appropriate band codes based on simulation sampling rates
 * and generates proper channel codes for synthetic seismogram outputs.
 *
 * The class is used by the seismogram writer to ensure consistent naming across different
 * wavefield types (displacement, velocity, acceleration, pressure) and component orientations.
 *
 * @see specfem::io::seismogram_writer
 * @see specfem::wavefield::type
 *
 * @code
 * // Example: Create a channel generator for a 50 Hz simulation
 * std::string output_dir = "OUTPUT_FILES";
 * type_real dt = 0.02;  // 50 Hz sampling rate
 * specfem::io::ChannelGenerator generator(output_dir, dt);
 *
 * // Generate displacement seismogram filenames for a station
 * auto filenames = generator.get_station_filenames(
 *     "II", "ANMO", specfem::wavefield::type::displacement);
 *
 * // Result: filenames contains:
 * // "OUTPUT_FILES/II.ANMO.BXX.semd"
 * // "OUTPUT_FILES/II.ANMO.BXY.semd"
 * // "OUTPUT_FILES/II.ANMO.BXZ.semd"
 *
 * // Get the band code
 * std::string band = generator.get_band_code();  // Returns "B" for dt=0.02s
 * @endcode
 */
// clang-format on
class ChannelGenerator {

public:
  /**
   * @brief Constructs a ChannelGenerator with output directory and simulation
   * timestep.
   *
   * Initializes the generator and automatically computes the SEED band code
   * based on the simulation sampling rate. The band code follows FDSN
   * conventions and ranges from 'L' (long period, ~1 Hz) to 'F' (very high
   * frequency, ≥1000 Hz).
   *
   * @param output_folder Directory path for seismogram output files
   * @param timestep Simulation time step (dt) in seconds, used to determine
   * band code
   */
  ChannelGenerator(const std::string output_folder, const type_real timestep)
      : output_folder(output_folder), band_code(compute_band_code(timestep)),
        timestep(timestep) {}

  // clang-format off
  /**
   * @brief Generates SEED-compliant seismogram filenames for a given station.
   *
   * Creates standardized output filenames following the SEED naming convention:
   * `<network>.<station>.<channel>.<extension>` where the channel code is
   * automatically determined based on the simulation timestep and component orientation.
   *
   * For elastic media (displacement, velocity, acceleration):
   * - Generates three-component seismograms with X, Y, Z components
   * - TODO: Future support for other coordinate systems (NEZ, RTZ)
   *
   * For acoustic media (pressure):
   * - Generates single-component seismogram with P component
   *
   * @param network_name SEED network code (e.g., "II", "IU", "SY")
   * @param station_name SEED station code (e.g., "ANMO", "STA01")
   * @param seismogram_type Wavefield type to output (displacement, velocity, acceleration, or pressure)
   * @return std::vector<std::string> Vector containing 3 filenames for elastic media or 1 for acoustic media
   *
   * @throws std::runtime_error If seismogram_type is not supported
   *
   * @code
   * // Example 1: Generate velocity seismogram filenames (100 Hz, band H)
   * specfem::io::ChannelGenerator gen("OUTPUT_FILES", 0.01);
   * auto files = gen.get_station_filenames("SY", "STA01", specfem::wavefield::type::velocity);
   * // Returns: ["OUTPUT_FILES/SY.STA01.HXX.semv",
   * //           "OUTPUT_FILES/SY.STA01.HXY.semv",
   * //           "OUTPUT_FILES/SY.STA01.HXZ.semv"]
   *
   * // Example 2: Generate pressure seismogram filename (acoustic)
   * auto press_files = gen.get_station_filenames("AC", "HYD01", specfem::wavefield::type::pressure);
   * // Returns: ["OUTPUT_FILES/AC.HYD01.HXP.semp"]
   * @endcode
   *
   * @see https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/
   */
  // clang-format on
  std::vector<std::string>
  get_station_filenames(const std::string &network_name,
                        const std::string &station_name,
                        const std::string &location_code,
                        specfem::wavefield::type seismogram_type);

  /**
   * @brief Returns the SEED band code for this generator.
   *
   * The band code is determined from the simulation timestep and follows FDSN
   * conventions for broad-band instruments. Valid codes are:
   * - 'L': Long Period (\f$ dt \geq 1.0 \f$ s, \f$ \approx 1 \f$ Hz)
   * - 'M': Mid Period (\f$ 0.1 < dt < 1.0 \f$ s, \f$ > 1 \f$ to \f$ < 10 \f$
   * Hz)
   * - 'B': Broad Band (\f$ 0.0125 < dt \leq 0.1 \f$ s, \f$ \geq 10 \f$ to \f$ <
   * 80 \f$ Hz)
   * - 'H': High Broad Band (\f$ 0.004 < dt \leq 0.0125 \f$ s, \f$ \geq 80 \f$
   * to \f$ < 250 \f$ Hz)
   * - 'C': Band C (\f$ 0.001 < dt \leq 0.004 \f$ s, \f$ \geq 250 \f$ to \f$ <
   * 1000 \f$ Hz)
   * - 'F': Band F (\f$ dt \leq 0.001 \f$ s, \f$ \geq 1000 \f$ Hz)
   *
   * @return std::string Single-character band code
   *
   * @see compute_band_code() for the mapping algorithm
   * @see https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/
   */
  std::string get_band_code() const { return this->band_code; }

  // clang-format off
  /**
   * @brief Generates a three-character SEED channel code for synthetic seismograms.
   *
   * Constructs a SEED channel code following the format: [Band][Instrument][Orientation]
   * - Band code: Determined from simulation timestep (L, M, B, H, C, or F)
   * - Instrument code: 'X' indicating derived/synthetic data
   * - Orientation code: Component direction (X, Y, Z, N, E, P, etc.)
   *
   * The 'X' instrument code denotes "Derived or Generated Channel" per SEED standards,
   * indicating this is synthetic data from numerical simulation rather than direct
   * instrumental recording.
   *
   * @param component_letter Single character specifying component orientation (e.g., 'X', 'Y', 'Z', 'P')
   * @return std::string Three-character SEED channel code (e.g., "BXZ", "HXN", "LXP")
   *
   * @code
   * specfem::io::ChannelGenerator gen("OUTPUT_FILES", 0.05);  // dt=0.05s → Band B
   * std::string channel_z = gen.get_channel_code('Z');  // Returns "BXZ"
   * std::string channel_n = gen.get_channel_code('N');  // Returns "BXN"
   * @endcode
   *
   * @see https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/#derived-or-generated-channel
   */
  // clang-format on
  std::string get_channel_code(const char component_letter);

  /**
   * @brief Returns the file extension for a given seismogram type.
   *
   * Maps wavefield types to standardized SPECFEM++ file extensions:
   * - displacement → "semd"
   * - velocity → "semv"
   * - acceleration → "sema"
   * - pressure → "semp"
   *
   * @param seismogram_type Type of wavefield output
   * @return std::string File extension without leading dot
   *
   * @throws std::runtime_error If seismogram_type is not supported
   */
  std::string get_file_extension(specfem::wavefield::type seismogram_type);

private:
  std::string output_folder;   ///< Directory path for seismogram output files
  const std::string band_code; ///< SEED band code (L, M, B, H, C, or F)
                               ///< determined from timestep
  type_real timestep;          ///< Simulation time step in seconds

  // clang-format off
  /**
   * @brief Computes the FDSN band code from simulation timestep.
   *
   * Determines the appropriate SEED band code based on sampling rate following
   * FDSN conventions for broad-band instruments. The band code indicates the
   * frequency range of the synthetic seismograms.
   *
   * FDSN Band Code Table (from IRIS SEED Appendix A):
   * =================================================
   *
   * | Band | Band Type                      | Sample Rate (Hz)        | Corner Period |
   * |------|--------------------------------|-------------------------|---------------|
   * | F    | ...                            | >= 1000 to < 5000       | >= 10 sec     |
   * | G    | ...                            | >= 1000 to < 5000       | < 10 sec      |
   * | D    | ...                            | >= 250 to < 1000        | < 10 sec      |
   * | C    | ...                            | >= 250 to < 1000        | >= 10 sec     |
   * | E    | Extremely Short Period         | >= 80 to < 250          | < 10 sec      |
   * | S    | Short Period                   | >= 10 to < 80           | < 10 sec      |
   * | H    | High Broad Band                | >= 80 to < 250          | >= 10 sec     |
   * | B    | Broad Band                     | >= 10 to < 80           | >= 10 sec     |
   * | M    | Mid Period                     | > 1 to < 10             | -             |
   * | L    | Long Period                    | ~= 1                    | -             |
   * | V    | Very Long Period               | ~= 0.1                  | -             |
   * | U    | Ultra Long Period              | ~= 0.01                 | -             |
   * | R    | Extremely Long Period          | >= 0.0001 to < 0.001    | -             |
   * | P    | On the order of 0.1 to 1 day   | >= 0.00001 to < 0.0001  | -             |
   * | T    | On the order of 1 to 10 days   | >= 0.000001 to < 0.00001| -             |
   * | Q    | Greater than 10 days           | < 0.000001              | -             |
   * | A    | Administrative Instrument Chan | variable                | NA            |
   * | O    | Opaque Instrument Channel      | variable                | NA            |
   *
   * Note: Sample rate in Hz = \f$ 1 / dt \f$ (sampling interval in seconds).
   * For example: \f$ dt = 0.01 \f$ s corresponds to sample rate = \f$ 100 \f$
   * Hz.
   *
   * SPECFEM++ follows the FDSN convention assuming broad-band characteristics
   * (corner period \f$ \geq 10 \f$ s) for consistency with observational
   * seismology.
   *
   * @param dt Simulation time step (sampling interval) in seconds
   * @return std::string Single-character band code
   *
   * @see https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/
   */
  // clang-format on
  static std::string compute_band_code(const type_real dt);
};

} // namespace specfem::io::impl
