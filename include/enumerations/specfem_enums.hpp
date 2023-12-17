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
  // primary boundaries
  none,                  ///< no boundary
  acoustic_free_surface, ///< free surface boundary for acoustic elements
  stacey,                ///< stacey boundary for elements

  // composite boundaries
  composite_stacey_dirichlet ///< composite boundary for acoustic elements
};

class boundary_tag_container {
public:
  std::vector<boundary_tag> get_tags() const { return tags; }

  boundary_tag_container(){};

  boundary_tag_container &operator=(const boundary_tag &tag) {
    if (tags.size() == 1 && tags[0] == boundary_tag::none) {
      tags[0] = tag;
    } else {
      tags.push_back(tag);
    }
    return *this;
  }

  bool operator==(const boundary_tag &tag) const {
    return (tags.size() == 1 && tags[0] == tag);
  }

  bool operator==(const std::tuple<boundary_tag, boundary_tag> &tag) const {
    return (tags.size() == 2 && tags[0] == std::get<0>(tag) &&
            tags[1] == std::get<1>(tag));
  }

  bool operator==(
      const std::tuple<boundary_tag, boundary_tag, boundary_tag> &tag) const {
    return (tags.size() == 3 && tags[0] == std::get<0>(tag) &&
            tags[1] == std::get<1>(tag) && tags[2] == std::get<2>(tag));
  }

private:
  std::vector<boundary_tag> tags = { boundary_tag::none };
};

constexpr bool operator==(const std::tuple<boundary_tag, boundary_tag> &lhs,
                          const boundary_tag &rhs) {
  return false;
}

constexpr bool operator==(const boundary_tag &lhs,
                          const std::tuple<boundary_tag, boundary_tag> &rhs) {
  return false;
}

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

namespace time_scheme {
/**
 * @brief type of time scheme
 *
 */
enum class type {
  newmark, ///< Newmark time scheme
};
} // namespace time_scheme
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_SPECFEM_ENUM_HPP_ */
