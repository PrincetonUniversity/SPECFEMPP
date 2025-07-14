#pragma once

#include "constants.hpp"
#include "enumerations/interface.hpp"
#include "io/writer.hpp"
#include "specfem/assembly.hpp"
#include "specfem_setup.hpp"
#include <vector>

namespace specfem {
namespace io {
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
  seismogram_writer(
      const specfem::enums::seismogram::format type,
      const specfem::enums::elastic_wave elastic_wave,
      const specfem::enums::electromagnetic_wave electromagnetic_wave,
      const std::string output_folder, const type_real dt, const type_real t0,
      const int nstep_between_samples)
      : type(type), elastic_wave(elastic_wave),
        electromagnetic_wave(electromagnetic_wave),
        output_folder(output_folder), dt(dt), t0(t0),
        nstep_between_samples(nstep_between_samples) {};
  /**
   * @brief Write seismograms
   *
   * @param assembly Assembly object
   *
   */
  void write(specfem::assembly::assembly<specfem::dimension::type::dim2>
                 &assembly) override;

private:
  specfem::enums::seismogram::format type; ///< Output format of the seismogram
                                           ///< file
  std::string output_folder; ///< Path to output folder where results will be
                             ///< stored
  type_real dt;              ///< Time interval between subsequent timesteps
  type_real t0;              ///< Solver start time
  specfem::enums::elastic_wave elastic_wave; ///< Type of wavefield (Writes
                                             ///< .BXY for SH waves and .BXX,
                                             ///< .BXZ for P-SV waves)
  specfem::enums::electromagnetic_wave
      electromagnetic_wave;  ///< Type of
                             ///< electromagnetic
                             ///< wavefield
  int nstep_between_samples; ///< number of timesteps between seismogram
                             ///< sampling (seismogram sampling frequency)
};

} // namespace io
} // namespace specfem
