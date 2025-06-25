#pragma once

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace io {

/**
 * @brief Writer to output wavefield data to disk
 *
 * @tparam OutputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename OutputLibrary> class wavefield_writer {

public:
  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Construct a writer object
   *
   * @param output_folder Path to output location (will be an .h5 file if using
   * HDF5, and a folder if using ASCII)
   */
  wavefield_writer(const std::string output_folder);
  ///@}

  /**
   * @brief Write the wavefield data to disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void write(specfem::compute::assembly &assembly);

  void write(specfem::compute::assembly &assembly, const int istep);

private:
  std::string output_folder; ///< Path to output folder
  typename OutputLibrary::File file;
};
} // namespace io
} // namespace specfem
