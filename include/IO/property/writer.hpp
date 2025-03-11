#pragma once

#include "IO/writer.hpp"
#include "compute/interface.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace IO {
/**
 * @brief Writer to model property data to disk
 *
 * @tparam OutputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename OutputLibrary> class property_writer : public writer {
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
  property_writer(const std::string output_folder);

  /**
   * @brief write the property data to disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void write(specfem::compute::assembly &assembly) override;

private:
  std::string output_folder; ///< Path to output folder
};
} // namespace IO
} // namespace specfem
