#ifndef _ENUMERATIONS_SPECFEM_ENUM_HPP_
#define _ENUMERATIONS_SPECFEM_ENUM_HPP_

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
/**
 * @namespace enums namespace is used to store enumerations.
 *
 */
namespace enums {

/**
 * @brief Cartesian axes
 *
 */
enum class axes {
  x, ///< X axis
  y, ///< Y axis
  z  ///< Z axis
};

namespace seismogram {
/**
 * @brief type of seismogram
 *
 */
enum class type {
  displacement, ///< Displacement seismogram
  velocity,     ///< Velocity Seismogram
  acceleration  ///< Acceleration seismogram
};

/**
 * @brief Output format of seismogram
 *
 */
enum format {
  seismic_unix, ///< Seismic unix output format
  ascii         ///< ASCII output format
};

} // namespace seismogram

/**
 * @namespace element namespace is used to store element properties used in the
 * element class.
 *
 */
namespace element {

/**
 * @brief type of element
 *
 * This is primarily used to label the element as elastic, acoustic or
 * poroelastic.
 *
 */
enum class type {
  elastic,    ///< elastic element
  acoustic,   ///< acoustic element
  poroelastic ///< poroelastic element
};

enum class property_tag {
  isotropic, ///< isotropic material
};

enum class boundary_tag {
  none,                  ///< no boundary
  acoustic_free_surface, ///< free surface boundary for acoustic elements
  stacey                 ///< stacey boundary for elastic elements
};

} // namespace element

namespace edge {
enum type {
  TOP,    ///< Top edge
  BOTTOM, ///< Bottom edge
  LEFT,   ///< Left edge
  RIGHT   ///< Right edge
};

constexpr int num_edges = 4; ///< Number of edges in the mesh
} // namespace edge

/**
 * @namespace boundaries enumeration namespace is used to store enumerations
 * used to describe various parts of the boundaries in a mesh.
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
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_SPECFEM_ENUM_HPP_ */
