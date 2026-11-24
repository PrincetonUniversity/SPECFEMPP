#pragma once

#include <string>

namespace specfem {
namespace enums {

/// Frame in which station coordinates are read and seismogram channels are
/// labelled.
enum class coordinate_frame {
  cartesian, ///< mesh axes: STATIONS give x/y/z, channels X/Y/Z
  geographic ///< lat/lon input; channels East/North/Up (E/N/Z)
};

/**
 * @brief Convert a coordinate frame to string.
 * @param frame Coordinate frame
 * @return "cartesian" or "geographic"
 */
const std::string to_string(const specfem::enums::coordinate_frame &frame);

} // namespace enums
} // namespace specfem
