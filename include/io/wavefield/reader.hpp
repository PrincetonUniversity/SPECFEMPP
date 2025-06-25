#pragma once

#include "compute/interface.hpp"
// #include "enumerations/interface.hpp"

namespace specfem {
namespace io {

/**
 * @brief Reader to read wavefield data from disk
 *
 */
template <typename IOLibrary> class wavefield_reader {

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
  void read(specfem::compute::assembly &assembly, const int istep);

private:
  std::string output_folder; ///< Path to output folder
};

} // namespace io
} // namespace specfem
