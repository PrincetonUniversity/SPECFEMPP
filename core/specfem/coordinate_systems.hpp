#pragma once

/**
 * @brief Umbrella header for coordinate types and projections.
 *
 * Includes geographic, cartesian, geocentric coordinate structs and all
 * available map projections (UTM, etc.).
 */

#include "coordinate_systems/cartesian.hpp"
#include "coordinate_systems/geocentric.hpp"
#include "coordinate_systems/geographic.hpp"
#include "coordinate_systems/input_coordinates.hpp"
#include "coordinate_systems/input_coordinates/input_cartesian_2d.hpp"
#include "coordinate_systems/input_coordinates/input_cartesian_3d.hpp"
#include "coordinate_systems/input_coordinates/input_cartesian_with_depth_3d.hpp"
#include "coordinate_systems/input_coordinates/input_geographic_3d.hpp"
#include "coordinate_systems/transform.hpp"
#include "coordinate_systems/utm.hpp"
