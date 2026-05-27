#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"
#include <array>
#include <optional>
#include <string>

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Cartesian coordinates in meters, templated by dimension.
 *
 * Carries an optional **origin** that controls how the coordinates are
 * resolved to mesh-space global coordinates:
 *
 * - `origin` has a value (e.g. `{0,0,0}`): the coordinates are absolute.
 *   Global position = stored values + origin.
 * - `origin` is `nullopt`: the coordinates require depth-based resolution.
 *   The origin must be set (e.g. from topography) before resolution.
 *   For the flat-topography fallback, the origin is set to `{0,...,0}`.
 *
 * Double precision is required for UTM round-trip accuracy (~1 mm).
 *
 * @tparam DimensionTag The dimension specification (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
class cartesian_coordinates;

//-------------------------- 2D Specialization --------------------------------

template <>
class cartesian_coordinates<specfem::element::dimension_tag::dim2> final
    : public coordinates<specfem::element::dimension_tag::dim2> {
public:
  double x = 0.0; ///< meters
  double z = 0.0; ///< meters (absolute z, or -depth if origin not set)
  std::optional<std::array<double, 2>> origin; ///< {0,0} = absolute; nullopt =
                                               ///< needs resolution

  /**
   * @brief Construct 2D Cartesian coordinates.
   *
   * @param x X coordinate in meters
   * @param z Z coordinate in meters (or -depth if depth-based)
   * @param origin Optional origin offset. Nullopt means depth-based.
   */
  cartesian_coordinates(double x, double z,
                        std::optional<std::array<double, 2>> origin =
                            std::array<double, 2>{ 0.0, 0.0 })
      : x(x), z(z), origin(origin) {}

  cartesian_coordinates() = default;

  bool operator==(const coordinates<specfem::element::dimension_tag::dim2>
                      &other) const override;
  std::string print() const override;
};

//-------------------------- 3D Specialization --------------------------------

template <>
class cartesian_coordinates<specfem::element::dimension_tag::dim3> final
    : public coordinates<specfem::element::dimension_tag::dim3> {
public:
  double x = 0.0; ///< meters (easting in UTM)
  double y = 0.0; ///< meters (northing in UTM)
  double z = 0.0; ///< meters (absolute z, or -depth if origin not set)
  std::optional<std::array<double, 3>> origin; ///< {0,0,0} = absolute; nullopt
                                               ///< = needs resolution

  /**
   * @brief Construct 3D Cartesian coordinates.
   *
   * @param x X coordinate in meters
   * @param y Y coordinate in meters
   * @param z Z coordinate in meters (or -depth if depth-based)
   * @param origin Optional origin offset. Nullopt means depth-based.
   */
  cartesian_coordinates(double x, double y, double z,
                        std::optional<std::array<double, 3>> origin =
                            std::array<double, 3>{ 0.0, 0.0, 0.0 })
      : x(x), y(y), z(z), origin(origin) {}

  cartesian_coordinates() = default;

  bool operator==(const coordinates<specfem::element::dimension_tag::dim3>
                      &other) const override;
  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
