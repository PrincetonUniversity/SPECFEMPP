#pragma once

#include "compute/interface.hpp"
// #include "enumerations/interface.hpp"
#include "reader/reader.hpp"

namespace specfem {
namespace reader {

/**
 * @brief Reader to read wavefield data from disk
 *
 */
template <typename IOLibrary> class wavefield : public reader {

public:
  /**
   * @brief Construct a new reader object
   *
   * @param output_folder Path to output folder or .h5 file
   */
  wavefield(const std::string &output_folder);

  /**
   * @brief Read the wavefield data from disk
   *
   * @param assembly SPECFEM++ assembly
   *
   */
  void read(specfem::compute::assembly &assembly) override;

private:
  std::string output_folder; ///< Path to output folder
};

} // namespace reader
} // namespace specfem
