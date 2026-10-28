#pragma once

#include "specfem/coordinate_systems/utm_projection.hpp"
#include "specfem/enums/coordinate_frame.hpp"
#include "specfem/mesh/mesh_base.hpp"

#include <variant>

namespace specfem::assembly::mesh_impl {

/**
 * @brief Pass-through projection: mesh coordinates are already Cartesian.
 *
 * Used for a plain Cartesian3D mesh (or a regional mesh with UTM suppressed).
 * Non-Cartesian inputs have no projection and are rejected by the conversion.
 */
struct cartesian_projection {};

/**
 * @brief Spherical-globe projection parameters.
 *
 * Carries the data the globe coordinate conversions read from the raw globe
 * mesh. @ref r_planet is the reference surface radius
 * (@ref specfem::globe::PlanetConstants::r_planet); a
 * @ref specfem::coordinate_systems::geocentric_projection_config is built from
 * it per call. @ref ellipticity / @ref topography gate the not-yet-implemented
 * deformed-surface resolution (issue #2058).
 */
struct globe_projection {
  double r_planet = 0.0;    ///< Reference surface radius in meters
  bool ellipticity = false; ///< True when the planet figure is elliptical
  bool topography = false;  ///< True when the mesh carries surface topography
};

/**
 * @brief The coordinate projection an assembled mesh was built with.
 *
 * Holds exactly one of: pass-through (@ref cartesian_projection), regional UTM
 * (@ref specfem::coordinate_systems::utm_projection_config), or spherical globe
 * (@ref globe_projection) -- never a mix. The source/receiver placement
 * conversions (@ref specfem::assembly::to_cartesian and its inverses) dispatch
 * on the active alternative, so the assembled mesh no longer needs the raw mesh
 * to resolve a coordinate. Default constructs to pass-through.
 */
struct coordinate_projection {
  std::variant<cartesian_projection,
               specfem::coordinate_systems::utm_projection_config,
               globe_projection>
      value{ cartesian_projection{} };

  coordinate_projection() = default;

  /**
   * @brief Build from a regional mesh: UTM when projection is active, else
   * pass-through (coordinates are already Cartesian).
   *
   * @param mesh Raw regional mesh carrying the UTM zone and suppress flag
   */
  explicit coordinate_projection(const specfem::mesh::cartesian3d_mesh &mesh);

  /**
   * @brief Build from a globe mesh: spherical projection from the planet
   * constants.
   *
   * @param mesh Raw globe mesh
   * @throws std::runtime_error if the mesh has no planet constants
   */
  explicit coordinate_projection(const specfem::mesh::globe3d_mesh &mesh);

  /**
   * @brief Frame in which stations are placed and seismogram channels labelled.
   *
   * @return `cartesian` for a pass-through projection; `geographic` for UTM or
   * globe
   */
  specfem::enums::coordinate_frame frame() const;
};

} // namespace specfem::assembly::mesh_impl
