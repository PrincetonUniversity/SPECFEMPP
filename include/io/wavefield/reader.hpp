#pragma once

#include "specfem/assembly.hpp"
// #include "enumerations/interface.hpp"

namespace specfem {
namespace io {

/**
 * @brief Reader to read wavefield data from disk
 *
 */
template <typename IOLibrary> class wavefield_reader {

public:
  /**
   * @brief Construct a new reader object
   *
   * @param output_folder Path to output folder or .h5 file
   */
  wavefield_reader(const std::string &output_folder);

  void initialize(
      specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly);

  /**
   * @brief Read the wavefield data from disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void
  run(specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
      const int istep);

  void finalize(
      specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {}

private:
  std::string output_folder;     ///< Path to output folder
  typename IOLibrary::File file; ///< File object to read from
};

} // namespace io
} // namespace specfem
