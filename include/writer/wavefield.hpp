#pragma once

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "writer/writer.hpp"

namespace specfem {
namespace writer {

/**
 * @brief Writer to output wavefield data to disk
 *
 * @tparam OutputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename OutputLibrary> class wavefield : public writer {

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
  wavefield(const specfem::compute::assembly &assembly,
            const std::string output_folder);
  ///@}

  /**
   * @brief Write the wavefield data to disk
   *
   */
  void write() override;

private:
  std::string output_folder; ///< Path to output folder
  specfem::compute::simulation_field<specfem::wavefield::type::forward>
      forward;                                       ///< Forward wavefield
  specfem::compute::boundary_values boundary_values; ///< Boundary values used
                                                     ///< for backward
                                                     ///< reconstruction during
                                                     ///< adjoint simulations
};
} // namespace writer
} // namespace specfem
