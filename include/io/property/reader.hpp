#pragma once

#include "enumerations/interface.hpp"
#include "io/reader.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace io {
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
  void read(specfem::assembly::assembly &assembly) override;

private:
  std::string input_folder;                 ///< Path to output folder
  specfem::assembly::properties properties; ///< Properties object
};
} // namespace io
} // namespace specfem
