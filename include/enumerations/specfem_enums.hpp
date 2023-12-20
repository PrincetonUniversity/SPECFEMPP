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

/**
 * @brief Container class to store boundary tags
 *
 *
 */
class boundary_tag_container {
public:
  /**
   * @brief Get the tags object
   *
   * @return std::vector<boundary_tag> vector of boundary tags
   */
  inline boundary_tag get_tag() const { return tag; }

  /**
   * @brief Construct a new boundary tag container object
   *
   */
  boundary_tag_container(){};

  boundary_tag_container &operator=(const boundary_tag &tag) = delete;

  /**
   * @brief Update boundary tag container with new tag
   *
   * This function checks if a boundary can be of composite type and returns the
   * correct tags
   *
   * @param rtag boundary tag to be added
   */
  boundary_tag_container &operator+=(const boundary_tag &rtag) {
    switch (rtag) {
    case boundary_tag::none:
      break;
    case boundary_tag::acoustic_free_surface:
      switch (this->tag) {
      case boundary_tag::none:
        this->tag = rtag;
        break;
      case boundary_tag::acoustic_free_surface:
      case boundary_tag::composite_stacey_dirichlet:
        break;
      case boundary_tag::stacey:
        this->tag = boundary_tag::composite_stacey_dirichlet;
        break;
      default:
        throw std::runtime_error("Invalid boundary tag");
        break;
      }
      break;
    case boundary_tag::stacey:
      switch (this->tag) {
      case boundary_tag::none:
        this->tag = rtag;
        break;
      case boundary_tag::acoustic_free_surface:
        this->tag = boundary_tag::composite_stacey_dirichlet;
        break;
      case boundary_tag::stacey:
      case boundary_tag::composite_stacey_dirichlet:
        break;
      default:
        throw std::runtime_error("Invalid boundary tag");
        break;
      }
      break;
    case boundary_tag::composite_stacey_dirichlet:
      switch (this->tag) {
      case boundary_tag::none:
        this->tag = rtag;
        break;
      case boundary_tag::acoustic_free_surface:
      case boundary_tag::stacey:
      case boundary_tag::composite_stacey_dirichlet:
        break;
      default:
        throw std::runtime_error("Invalid boundary tag");
        break;
      }
      break;
    default:
      throw std::runtime_error("Invalid boundary tag");
      break;
    }

    return *this;
  }

  /**
   * @brief Check if boundary tag container specifies a specific boundary tag
   *
   * This function checks if a boundary container specifies a specific boundary
   * tag
   *
   * @param tag boundary tag to be checked
   * @return bool true if boundary container specifies the boundary tag
   */
  bool operator==(const boundary_tag &tag) const { return (tag == this->tag); }

private:
  boundary_tag tag = boundary_tag::none; ///< boundary tag
};

} // namespace element

/**
 * @namespace edge namespace is used to store enumerations used to describe the
 * edges
 *
 */
namespace edge {
/**
 * @brief type of edge in the mesh
 *
 */
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
