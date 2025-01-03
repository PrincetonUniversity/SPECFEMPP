#pragma once

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "writer/writer.hpp"

namespace specfem {
namespace writer {

/**
 * @brief Writer to output wavefield data to disk
 *
 * @tparam OutputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename OutputLibrary> class wavefield : public writer {

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
  wavefield(const std::string output_folder);
  ///@}

  /**
   * @brief Write the wavefield data to disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void write(specfem::compute::assembly &assembly) override;

private:
  std::string output_folder; ///< Path to output folder
};
} // namespace writer
} // namespace specfem
