#pragma once

#include "specfem/ellipticity/tags.hpp"

namespace specfem {
namespace ellipticity {

/**
 * @brief Compile-time ellipsoid parameters (semi-major axis @f$ a @f$,
 *        semi-minor axis @f$ b @f$) in meters.
 *
 * Specialize for each model in the @ref model enum. All values are `double`.
 *
 * @code
 * using namespace specfem::ellipticity;
 * constexpr auto wgs = ellipsoid<model::wgs84>{};
 * double a = wgs.semi_major_axis;  // 6378137.0
 * double b = wgs.semi_minor_axis;  // 6356752.314245
 * @endcode
 *
 * @tparam Model The ellipsoid model tag.
 */
template <model Model> struct ellipsoid;

/// @brief Clarke 1866 ellipsoid (NAD27): @f$ a = 6378206.4 @f$ m.
template <> struct ellipsoid<model::clarke_1866> {
  static constexpr double semi_major_axis = 6378206.4;
  static constexpr double semi_minor_axis = 6356583.8;
};

/// @brief WGS-84 ellipsoid: @f$ a = 6378137.0 @f$ m, @f$ b = 6356752.314245 @f$
/// m.
template <> struct ellipsoid<model::wgs84> {
  static constexpr double semi_major_axis = 6378137.0;
  static constexpr double semi_minor_axis = 6356752.314245;
};

} // namespace ellipticity
} // namespace specfem
