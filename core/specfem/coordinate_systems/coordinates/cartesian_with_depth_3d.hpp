#pragma once

#include "specfem/coordinate_systems/coordinates.hpp"

namespace specfem {
namespace coordinate_systems {

/**
 * @brief 3D Cartesian coordinates with depth (x, y, depth) in meters.
 *
 * Depth is measured positive downward from the topographic surface.
 * Conversion to @ref specfem::point::global_coordinates requires
 * knowledge of surface topography (available at assembly time).
 */
class cartesian_with_depth_3d final
    : public coordinates<specfem::element::dimension_tag::dim3> {
public:
  double x;     ///< meters (easting in UTM)
  double y;     ///< meters (northing in UTM)
  double depth; ///< meters (positive down from topographic surface)

  cartesian_with_depth_3d(double x, double y, double depth)
      : x(x), y(y), depth(depth) {}

  std::string print() const override;
};

} // namespace coordinate_systems
} // namespace specfem
