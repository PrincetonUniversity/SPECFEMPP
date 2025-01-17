#pragma once

#include "IO/writer.hpp"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "receiver/interface.hpp"
#include "specfem_setup.hpp"
#include <vector>

namespace specfem {
namespace IO {
/**
 * @brief Seismogram writer class to write seismogram to a file
 *
 */
class seismogram_writer : public writer {

public:
  /**
   * @brief Construct a new seismogram writer object
   *
   * @param type Format of the output file
   * @param output_folder path to output folder where results will be stored
   * @param dt Time interval between subsequent timesteps
   * @param t0 Solver start time
   * @param nstep_between_samples number of timesteps between seismogram
   * sampling (seismogram sampling frequency)
   */
  seismogram_writer(const specfem::enums::seismogram::format type,
                    const std::string output_folder, const type_real dt,
                    const type_real t0, const int nstep_between_samples)
      : type(type), output_folder(output_folder), dt(dt), t0(t0),
        nstep_between_samples(nstep_between_samples){};
  /**
   * @brief Write seismograms
   *
   * @param assembly Assembly object
   *
   */
  void write(specfem::compute::assembly &assembly) override;

private:
  specfem::enums::seismogram::format type; ///< Output format of the seismogram
                                           ///< file
  std::string output_folder; ///< Path to output folder where results will be
                             ///< stored
  type_real dt;              ///< Time interval between subsequent timesteps
  type_real t0;              ///< Solver start time
  int nstep_between_samples; ///< number of timesteps between seismogram
                             ///< sampling (seismogram sampling frequency)
};

} // namespace IO
} // namespace specfem
