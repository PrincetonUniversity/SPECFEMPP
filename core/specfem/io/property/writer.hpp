#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/writer.hpp"

namespace specfem {
namespace io {
/**
 * @brief Writer for outputting material properties to disk
 *
 * Template-based writer for material property data supporting multiple I/O
 * backends. Used to write density, velocities, and other material parameters.
 *
 * The file holds one group per (medium, property, attenuation) combination,
 * named specfem::element::to_string(medium, property, attenuation); property
 * datasets cover only that combination's elements. For attenuating
 * combinations the "kappa"/"mu" datasets hold the reference
 * (physical/relaxed) moduli owned by the attenuation container -- persisted
 * verbatim alongside the attenuation model datasets (e.g. Qkappa/Qmu) --
 * rather than the runtime (unrelaxed) values. Requires an assembly
 * constructed with property I/O enabled.
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
  ///@}

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
                 &assembly) override;

private:
  std::string output_folder; ///< Path to output folder
};
} // namespace io
} // namespace specfem
