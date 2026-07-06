#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"

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
   * @tparam DimensionTag Spatial dimension (dim2 or dim3)
   * @param assembly SPECFEM++ assembly
   *
   */
  template <specfem::element::dimension_tag DimensionTag>
  void initialize(specfem::assembly::assembly<DimensionTag> &assembly);

  template <specfem::element::dimension_tag DimensionTag>
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep);

  /**
   * @brief Write wavefield snapshot including attenuation memory variables
   *
   * Used by the UNDO_ATTENUATION solver to checkpoint the complete simulation
   * state at subset boundaries. Saves displacement/velocity/acceleration and
   * the SLS memory variables (Rxx, Rxz, Rkappa for dim2; Rxx, Ryy, Rxy, Rxz,
   * Ryz, Rkappa for dim3) needed to restart the forward attenuation physics
   * from this point. Strain (epsilondev) is NOT saved and will be recomputed
   * from the displacement gradient, matching the Fortran strategy.
   *
   * @tparam DimensionTag Spatial dimension (dim2 or dim3)
   * @param assembly SPECFEM++ assembly
   * @param istep    Current time step index used to name the on-disk group
   */
  template <specfem::element::dimension_tag DimensionTag>
  void run_with_attenuation(specfem::assembly::assembly<DimensionTag> &assembly,
                            const int istep);

  template <specfem::element::dimension_tag DimensionTag>
  void finalize(specfem::assembly::assembly<DimensionTag> &assembly);

private:
  std::string output_folder; ///< Path to output folder
  std::string file_path; ///< Rank-specific path to the wavefield file/folder
  typename OutputLibrary::File file;
  bool save_boundary_values;
};
} // namespace io
} // namespace specfem

#include "specfem/io/wavefield/writer.tpp"
