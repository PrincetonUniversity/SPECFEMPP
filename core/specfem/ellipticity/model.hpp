#pragma once

namespace specfem {
namespace ellipticity {

/**
 * @brief Available reference ellipsoid models.
 *
 * Used as a template parameter to @ref ellipsoid to select axis values.
 */
enum class model {
  clarke_1866, ///< Clarke 1866 (NAD27)
  wgs84        ///< WGS-84 (World Geodetic System 1984, default)
};

} // namespace ellipticity
} // namespace specfem
