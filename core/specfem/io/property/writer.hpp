#pragma once

#include "specfem/assembly/fwd.hpp"
#include "specfem/io/writer.hpp"

namespace specfem {
namespace io {
/**
 * @brief Writer for outputting material properties to disk
 *
 * Template-based writer for material property data supporting multiple I/O
 * backends. Used to write density, velocities, and other material parameters.
 *
 * @tparam OutputLibrary Backend library type (HDF5, ASCII, NPY, NPZ, or ADIOS2)
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
  property_writer(const std::string &output_folder);

  /**
   * @brief write the property data to disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void write(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                 &assembly) override;

  /**
   * @brief write the property data to disk
   *
   * @param assembly SPECFEM++ 3D assembly
   *
   */
  void write(specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
                 &assembly) override {
    throw std::runtime_error("3D property writing not yet implemented");
  };

private:
  std::string output_folder; ///< Path to output folder
};
} // namespace io
} // namespace specfem
