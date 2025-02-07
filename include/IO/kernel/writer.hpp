#pragma once

#include "IO/writer.hpp"
#include "compute/interface.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace IO {
/**
 * @brief Writer to output misfit kernel data to disk
 *
 * @tparam OutputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename OutputLibrary> class kernel_writer : public writer {
public:
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a writer object
   *
   * @param assembly SPECFEM++ assembly
   * @param output_folder Path to output location (will be an .h5 file if using
   * HDF5, and a folder if using ASCII)
   */
  kernel_writer(const std::string output_folder);

  /**
   * @brief write the kernel data to disk
   *
   */
  void write(specfem::compute::assembly &assembly) override;

private:
  std::string output_folder; ///< Path to output folder
};
} // namespace IO
} // namespace specfem
