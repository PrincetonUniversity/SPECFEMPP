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

  /**
   * @brief Read wavefield snapshot including attenuation memory variables
   *
   * Counterpart to wavefield_writer::run_with_attenuation. Reads
   * displacement/velocity/acceleration into assembly.fields.buffer and the
   * SLS memory variables into assembly.attenuation (which the
   * combined_undoatt solver will then deep_copy into the forward attenuation
   * container before replaying the subset).
   *
   * @tparam DimensionTag Spatial dimension (dim2 or dim3)
   * @param assembly SPECFEM++ assembly
   * @param istep    Time step index of the checkpoint to load
   */
  template <specfem::element::dimension_tag DimensionTag>
  void run_with_attenuation(specfem::assembly::assembly<DimensionTag> &assembly,
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
};

} // namespace io
} // namespace specfem

#include "specfem/io/wavefield/reader.tpp"
