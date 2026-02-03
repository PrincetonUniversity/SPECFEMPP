#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace io {

/**
 * @brief Writer for outputting wavefield data to disk
 *
 * Template-based writer supporting multiple I/O backends. Saves displacement,
 * velocity, and acceleration fields at specified time steps. Can optionally
 * save boundary values for domain decomposition interfaces.
 *
 * @tparam OutputLibrary Backend library type (HDF5, ASCII, NPY, NPZ, or ADIOS2)
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
  wavefield_writer(const std::string &output_folder,
                   const bool save_boundary_values);
  ///@}

  /**
   * @brief Write the wavefield data to disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void initialize(
      specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly);

  void
  run(specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
      const int istep);

  void finalize(
      specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly);

private:
  std::string output_folder; ///< Path to output folder
  typename OutputLibrary::File file;
  bool save_boundary_values;
};
} // namespace io
} // namespace specfem
