#pragma once

#include "specfem/io.hpp"
#include "specfem/setup.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <optional>
#include <tuple>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Seismogram class is used to instantiate seismogram writer
 *
 */
class seismogram {

public:
  /**
   * @brief Construct a new seismogram object
   *
   * @param output_format Outpul seismogram file format
   * @param output_folder Path to folder location where seismogram will be
   * stored
   * @param write_from_main Gather all seismogram data to rank 0 and write only
   * from the main process
   */
  seismogram(const std::string &output_format, const std::string &output_folder,
             const std::optional<bool> write_from_main)
      : output_format(output_format), output_folder(output_folder),
        write_from_main(write_from_main) {};
  /**
   * @brief Construct a new seismogram object
   *
   * @param Node YAML node describing the seismogram writer
   */
  seismogram(const YAML::Node &Node);

  /**
   * @brief Instantiate a seismogram writer object
   *
   * @param wave_type Type of wavefield (Writes .BXY for SH waves and .BXX, .BXZ
   * for P-SV waves)
   * @param dt Time interval between subsequent timesteps
   * @param t0 Solver start time
   * @param nsteps_between_samples number of timesteps between seismogram
   * sampling (seismogram sampling frequency)
   * @return std::shared_ptr<specfem::io::writer> Pointer to an instantiated
   * writer object
   */
  std::shared_ptr<specfem::io::writer> instantiate_seismogram_writer(
      const specfem::enums::elastic_wave wave_type,
      const specfem::enums::electromagnetic_wave electromagnetic_wave,
      const type_real dt, const type_real t0,
      const int nsteps_between_samples) const;

  /**
   * @brief Whether to gather all seismogram data to rank 0 and write only from
   * the main process
   */
  std::optional<bool> get_write_from_main() const { return write_from_main; }

private:
  std::string output_format; ///< format of output file
  std::string output_folder; ///< Path to output folder
  std::optional<bool> write_from_main =
      std::nullopt; ///< Gather to rank 0 and write from main proc
};

} // namespace runtime_configuration
} // namespace specfem
