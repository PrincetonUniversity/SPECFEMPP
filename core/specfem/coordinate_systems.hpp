#pragma once

/**
 * @brief Umbrella header for coordinate types and projections.
 *
 * Includes geographic, cartesian, geocentric coordinate structs and all
 * available map projections (UTM, etc.).
 */

#include "coordinate_systems/cartesian.hpp"
#include "coordinate_systems/coordinates.hpp"
#include "coordinate_systems/coordinates/cartesian_2d.hpp"
#include "coordinate_systems/coordinates/cartesian_3d.hpp"
#include "coordinate_systems/coordinates/cartesian_with_depth_3d.hpp"
#include "coordinate_systems/coordinates/geographic_3d.hpp"
#include "coordinate_systems/geocentric.hpp"
#include "coordinate_systems/geographic.hpp"
#include "coordinate_systems/transform.hpp"
#include "coordinate_systems/utm.hpp"
