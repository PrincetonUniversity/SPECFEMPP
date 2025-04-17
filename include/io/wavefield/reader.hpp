#pragma once

#include "compute/interface.hpp"
// #include "enumerations/interface.hpp"
#include "io/reader.hpp"

namespace specfem {
namespace io {

/**
 * @brief Reader to read wavefield data from disk
 *
 */
template <typename IOLibrary> class wavefield_reader : public reader {

public:
  /**
   * @brief Construct a new reader object
   *
   * @param output_folder Path to output folder or .h5 file
   */
  wavefield_reader(const std::string &output_folder);

  /**
   * @brief Read the wavefield data from disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void read(specfem::compute::assembly &assembly) override;

  void set_istep(int istep) { this->istep = istep; }

private:
  std::string output_folder; ///< Path to output folder
  int istep = -1; ///< Current time step for file name, if value is -1, the time
                  ///< step will be be included in the output file name.
};

} // namespace io
} // namespace specfem
