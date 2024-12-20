#pragma once

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "writer/writer.hpp"

namespace specfem {
namespace writer {
/**
 * @brief Writer to model property data to disk
 *
 * @tparam OutputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename OutputLibrary> class property : public writer {
public:
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a writer object
   *
   * @param assembly SPECFEM++ assembly
   * @param output_folder Path to output location (will be an .h5 file if using
   * HDF5, and a folder if using ASCII)
   */
  property(const specfem::compute::assembly &assembly,
         const std::string output_folder);

  /**
   * @brief write the property data to disk
   *
   */
  void write() override;

private:
  std::string output_folder;         ///< Path to output folder
  specfem::compute::mesh mesh;       ///< Mesh object
  specfem::compute::properties properties; ///< Properties object
};
} // namespace writer
} // namespace specfem
