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
   * @param save_boundary_values Whether to checkpoint Stacey/composite boundary
   * values in finalize()
   * @param save_attenuation_value Whether attenuation is enabled; when true,
   * run() also checkpoints the SLS memory variables alongside the kinematic
   * fields
   */
  wavefield_writer(const std::string &output_folder,
                   const bool save_boundary_values,
                   const bool save_attenuation_value);
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

  /**
   * @brief Write the wavefield snapshot for a single time step
   *
   * Saves displacement/velocity/acceleration (or the acoustic potentials) for
   * every medium. When the writer was constructed with
   * @p save_attenuation_value set, this additionally checkpoints the SLS memory
   * variables (Rxx, Rxz, Rkappa for dim2; Rxx, Ryy, Rxy, Rxz, Ryz, Rkappa for
   * dim3) needed by the UNDO_ATTENUATION solver to restart the forward
   * attenuation physics from this point. Strain (epsilondev) is NOT saved and
   * will be recomputed from the displacement gradient, matching the Fortran
   * strategy.
   *
   * @tparam DimensionTag Spatial dimension (dim2 or dim3)
   * @param assembly SPECFEM++ assembly
   * @param istep    Current time step index used to name the on-disk group
   */
  template <specfem::element::dimension_tag DimensionTag>
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep);

  template <specfem::element::dimension_tag DimensionTag>
  void finalize(specfem::assembly::assembly<DimensionTag> &assembly);

private:
  std::string output_folder; ///< Path to output folder
  std::string file_path; ///< Rank-specific path to the wavefield file/folder
  typename OutputLibrary::File file;
  bool save_boundary_values;
  bool save_attenuation_value; ///< Whether to checkpoint attenuation memory
                               ///< variables alongside the kinematic fields
};
} // namespace io
} // namespace specfem

#include "specfem/io/wavefield/writer.tpp"
