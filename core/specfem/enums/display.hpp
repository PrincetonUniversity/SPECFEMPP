#pragma once
#include <string>

namespace specfem {
namespace enums {

/**
 * @brief Output formats for wavefield visualization and data export.
 *
 * Controls how simulation results are rendered and saved. Used by
 * wavefield plotting functions to determine output method.
 */
enum class display_format {
  PNG,       ///< PNG image output (2D only)
  JPG,       ///< JPEG image output (2D only)
  on_screen, ///< Interactive display (2D only)
  vtkhdf     ///< VTK HDF5 format (2D and 3D)
};

enum class display_component { x, y, z, magnitude };

std::string to_string(const display_format &fmt);
std::string to_string(const display_component &comp);

} // namespace enums
} // namespace specfem
