#include "specfem/assembly/mesh/dim3/impl/coordinate_projection.hpp"

#include "specfem/globe/planet_constants.hpp"
#include "specfem/mesh.hpp"

#include <stdexcept>

// Regional mesh: UTM when projection is active, else pass-through (coordinates
// are already Cartesian).
specfem::assembly::mesh_impl::coordinate_projection::coordinate_projection(
    const specfem::mesh::cartesian3d_mesh &mesh) {
  if (mesh.suppress_utm_projection) {
    value = cartesian_projection{};
  } else {
    value = specfem::coordinate_systems::utm_projection_config{
      mesh.utm_projection_zone, false
    };
  }
}

// Globe mesh: spherical projection; the reference surface radius comes from the
// planet constants read with the thin database.
specfem::assembly::mesh_impl::coordinate_projection::coordinate_projection(
    const specfem::mesh::globe3d_mesh &mesh) {
  if (!mesh.globe.planet_constants.has_value()) {
    throw std::runtime_error(
        "specfem::assembly::mesh_impl::coordinate_projection: globe mesh is "
        "missing the planet constants required for coordinate projection");
  }
  value = globe_projection{ mesh.globe.planet_constants.value().r_planet(),
                            mesh.globe.model_config.ellipticity,
                            mesh.globe.model_config.topography };
}
