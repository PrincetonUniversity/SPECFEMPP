#pragma once

#include "specfem/assembly/properties.hpp"
#include "specfem/io/reader.hpp"

namespace specfem {
namespace io {
/**
 * @brief Reader for loading material properties from disk
 *
 * Template-based reader for material property data supporting multiple I/O
 * backends. Used to read density, velocities, and other material parameters.
 *
 * @tparam InputLibrary Backend library type (HDF5, ASCII, NPY, NPZ, or ADIOS2)
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
  property_reader(const std::string &input_folder);

  /**
   * @brief read the property from disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void read(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                &assembly) override;

private:
  std::string input_folder; ///< Path to output folder
  specfem::assembly::properties<specfem::element::dimension_tag::dim2>
      properties; ///< Properties object
};
} // namespace io
} // namespace specfem
