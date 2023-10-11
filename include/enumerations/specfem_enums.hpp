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
enum class type {
  displacement, ///< Displacement seismogram
  velocity,     ///< Velocity Seismogram
  acceleration  ///< Acceleration seismogram
};

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
enum type {
  elastic,    ///< elastic element
  acoustic,   ///< acoustic element
  poroelastic ///< poroelastic element
};

} // namespace element

namespace coupling {
namespace edge {
enum type {
  TOP,    ///< Top edge
  BOTTOM, ///< Bottom edge
  LEFT,   ///< Left edge
  RIGHT   ///< Right edge
};

constexpr int num_edges = 4; ///< Number of edges in the mesh
} // namespace edge
} // namespace coupling
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_SPECFEM_ENUM_HPP_ */
