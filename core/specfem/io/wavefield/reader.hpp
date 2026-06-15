#pragma once

#include "specfem/assembly/assembly.hpp"
// #include "specfem/enums.hpp"

namespace specfem {
namespace io {

/**
 * @brief Reader for loading wavefield data from disk
 *
 * Template-based reader supporting multiple I/O backends. Reads displacement,
 * velocity, and acceleration fields at specified time steps.
 *
 * @tparam IOLibrary Backend library type (HDF5, ASCII, NPY, NPZ, or ADIOS2)
 */
template <typename IOLibrary> class wavefield_reader {

public:
  /**
   * @brief Construct a new reader object
   *
   * @param output_folder Path to output folder or .h5 file
   */
  wavefield_reader(const std::string &output_folder);

  /**
   * @brief Read the wavefield data from disk
   *
   * @tparam DimensionTag Spatial dimension (dim2 or dim3)
   * @param assembly SPECFEM++ assembly
   *
   */
  template <specfem::element::dimension_tag DimensionTag>
  void initialize(specfem::assembly::assembly<DimensionTag> &assembly);

  template <specfem::element::dimension_tag DimensionTag>
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep);

  template <specfem::element::dimension_tag DimensionTag>
  void finalize(specfem::assembly::assembly<DimensionTag> &assembly) {}

private:
  std::string output_folder; ///< Path to output folder
  std::string file_path; ///< Rank-specific path to the wavefield file/folder
  typename IOLibrary::File file; ///< File object to read from
};

} // namespace io
} // namespace specfem

#include "specfem/io/wavefield/reader.tpp"
