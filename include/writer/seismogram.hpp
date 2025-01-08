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
   * @param output_folder Output folder to write seismograms
   * @param assembly Assembly object where recievers and seismograms are stored'
   * @param type Seismogram format
   */
  seismogram(const std::string output_folder,
             const specfem::compute::assembly &assembly,
             const specfem::enums::seismogram::format type)
      : receivers(assembly.receivers), output_folder(output_folder),
        type(type) {}
  /**
   * @brief Write seismograms
   *
   */
  void write() override;

private:
  specfem::compute::receivers receivers;
  std::string output_folder;
  specfem::enums::seismogram::format type;
};

} // namespace writer
} // namespace specfem

#endif
