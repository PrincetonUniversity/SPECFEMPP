#pragma once

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
/**
 * @namespace enums namespace is used to store enumerations.
 *
 */
namespace enums {

/**
 * @brief Cartesian axes enumeration
 *
 */
enum class axes { x, y, z };

namespace seismogram {
/**
 * @brief Seismogram type enumeration
 *
 */
enum class type { displacement, velocity, acceleration };

/**
 * @brief Output format of seismogram enumeration
 *
 */
enum format { seismic_unix, ascii };

} // namespace seismogram

/**
 * @namespace edge namespace is used to store enumerations used to describe the
 * edges
 *
 */
namespace edge {
/**
 * @brief Edge type enumeration
 *
 */
enum type {
  NONE = 0,   /// Not an edge
  TOP = 1,    ///< Top edge
  BOTTOM = 2, ///< Bottom edge
  LEFT = 3,   ///< Left edge
  RIGHT = 4   ///< Right edge
};

constexpr int num_edges = 5; ///< Number of edges in the mesh
} // namespace edge

/**
 * @namespace Boundaries enumeration
 *
 */
namespace boundaries {
/**
 * @brief type of the boundary (corner, edge)
 *
 */
enum type {
  TOP_LEFT,     ///< Top left corner
  TOP_RIGHT,    ///< Top right corner
  BOTTOM_LEFT,  ///< Bottom left corner
  BOTTOM_RIGHT, ///< Bottom right corner
  TOP,          ///< Top edge
  LEFT,         ///< Left edge
  RIGHT,        ///< Right edge
  BOTTOM        ///< Bottom edge
};
} // namespace boundaries

namespace time_scheme {
/**
 * @brief Time scheme enumeration
 *
 */
enum class type {
  newmark, ///< Newmark time scheme
};
} // namespace time_scheme
} // namespace enums
} // namespace specfem
