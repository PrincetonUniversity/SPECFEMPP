#ifndef _SEISMOGRAM_WRITER_HPP
#define _SEISMOGRAM_WRITER_HPP

#include "compute/interface.hpp"
#include "constants.hpp"
#include "receiver/interface.hpp"
#include "specfem_setup.hpp"
#include "writer.hpp"
#include <vector>

namespace specfem {
namespace writer {
/**
 * @brief Seismogram writer class to write seismogram to a file
 *
 */
class seismogram : public writer {

public:
  /**
   * @brief Construct a new seismogram writer object
   *
   * @param receivers Vector of pointers to specfem::receivers::receiver objects
   * @param compute_receivers Pointer to specfem::compute::receivers object
   * @param type Format of the output file
   * @param output_folder path to output folder where results will be stored
   * @param dt Time interval between subsequent timesteps
   * @param t0 Solver start time
   * @param nstep_between_samples number of timesteps between seismogram
   * sampling (seismogram sampling frequency)
   */
  seismogram(
      std ::vector<std::shared_ptr<specfem::receivers::receiver> > &receivers,
      specfem::compute::receivers &compute_receivers,
      const specfem::enums::seismogram::format type,
      const std::string output_folder, const type_real dt, const type_real t0,
      const int nstep_between_samples)
      : receivers(receivers), compute_receivers(compute_receivers), type(type),
        output_folder(output_folder), dt(dt), t0(t0),
        nstep_between_samples(nstep_between_samples){};
  /**
   * @brief Write seismograms
   *
   */
  void write() override;

private:
  specfem::enums::seismogram::format type; ///< Output format of the seismogram
                                           ///< file
  std::string output_folder; ///< Path to output folder where results will be
                             ///< stored
  specfem::compute::receivers
      compute_receivers; ///< Pointer to
                         ///< specfem::compute::receivers
                         ///< object. This object
                         ///< containes the view used
                         ///< to store calculated
                         ///< seismograms
  std::vector<std::shared_ptr<specfem::receivers::receiver> >
      receivers; ///< Vector of pointers to specfem::receivers::receiver
                 ///< objects. These are used to get station and network name
                 ///< where saving the seismogram
  type_real dt;  ///< Time interval between subsequent timesteps
  type_real t0;  ///< Solver start time
  int nstep_between_samples; ///< number of timesteps between seismogram
                             ///< sampling (seismogram sampling frequency)
};

} // namespace writer
} // namespace specfem

#endif
