#pragma once

#include "IO/reader.hpp"
#include "compute/interface.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace IO {
/**
 * @brief Read model property
 *
 * @tparam InputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename InputLibrary> class property_reader : public reader {
public:
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a reader object
   *
   * @param output_folder Path to input location (will be an .h5 file if using
   * HDF5, and a folder if using ASCII)
   */
  property_reader(const std::string input_folder);

  /**
   * @brief read the property from disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void read(specfem::compute::assembly &assembly) override;

private:
  std::string input_folder;                ///< Path to output folder
  specfem::compute::properties properties; ///< Properties object
};
} // namespace IO
} // namespace specfem
