#pragma once

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "reader/reader.hpp"

namespace specfem {
namespace reader {
/**
 * @brief Read model property
 *
 * @tparam InputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename InputLibrary> class property : public reader {
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
  property(const std::string input_folder);

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
} // namespace reader
} // namespace specfem
