#pragma once

#include "specfem/assembly/assembly.hpp"
#include <optional>
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
   * @param load_attenuation_value Whether attenuation is enabled; when true,
   * run() also loads the SLS memory variables into assembly.attenuation
   */
  wavefield_reader(const std::string &output_folder,
                   const bool load_attenuation_value);

  /**
   * @brief Read the wavefield data from disk
   *
   * @tparam DimensionTag Spatial dimension (dim2 or dim3)
   * @param assembly SPECFEM++ assembly
   *
   */
  template <specfem::element::dimension_tag DimensionTag>
  void initialize(specfem::assembly::assembly<DimensionTag> &assembly);

  /**
   * @brief Read the wavefield snapshot for a single time step
   *
   * Loads displacement/velocity/acceleration (or the acoustic potentials) into
   * assembly.fields.buffer. When the reader was constructed with
   * @p load_attenuation_value set, this additionally loads the SLS memory
   * variables into assembly.attenuation (which the combined_undoatt solver then
   * deep_copies into the forward attenuation container before replaying the
   * subset).
   *
   * @tparam DimensionTag Spatial dimension (dim2 or dim3)
   * @param assembly SPECFEM++ assembly
   * @param istep    Time step index of the checkpoint to load
   */
  template <specfem::element::dimension_tag DimensionTag>
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep);

  template <specfem::element::dimension_tag DimensionTag>
  void finalize(specfem::assembly::assembly<DimensionTag> &assembly) {}

  /**
   * @brief Open the backing file.
   *
   * For combined_undoatt workflows the checkpoint directory is created during
   * the forward pass, so the file cannot be opened at construction time.
   * Call this once before the first read (i.e. inside initialize()).
   */
  void open_file();

private:
  std::string output_folder; ///< Path to output folder
  std::string file_path; ///< Rank-specific path to the wavefield file/folder
  std::optional<typename IOLibrary::File> file; ///< Lazily-opened file object
  bool load_attenuation_value; ///< Whether to load attenuation memory variables
                               ///< alongside the kinematic fields
};

} // namespace io
} // namespace specfem

#include "specfem/io/wavefield/reader.tpp"
