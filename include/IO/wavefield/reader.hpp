#pragma once

#include "compute/interface.hpp"
// #include "enumerations/interface.hpp"
#include "IO/reader.hpp"

namespace specfem {
namespace IO {

/**
 * @brief Reader to read wavefield data from disk
 *
 */
template <typename IOLibrary> class wavefield_reader : public reader {

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
   * @param assembly SPECFEM++ assembly
   *
   */
  void read(specfem::compute::assembly &assembly) override;

private:
  std::string output_folder; ///< Path to output folder
};

} // namespace IO
} // namespace specfem
