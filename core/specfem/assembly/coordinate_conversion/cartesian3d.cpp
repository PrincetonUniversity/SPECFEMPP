#include "specfem/assembly/coordinate_conversion.hpp"

#include "specfem/algorithms/locate_point.hpp"
#include "specfem/assembly/coordinate_conversion/impl.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/to.hpp"
#include "specfem/coordinate_systems/utm_projection.hpp"
#include "specfem/mesh.hpp"
#include "specfem/setup.hpp"

#include <stdexcept>

// Shared Cartesian placement: absolute coordinates pass through; a depth-based
// coordinate (origin unset) is placed against the free surface above (x, y).
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::assembly::coordinate_conversion_impl::resolve_cartesian(
    const specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3> &cartesian,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::acoustic_free_surface<
        specfem::element::dimension_tag::dim3> &surface) {

  if (cartesian.origin.has_value()) {
    const auto &origin = *cartesian.origin;
    return { static_cast<type_real>(cartesian.x + origin[0]),
             static_cast<type_real>(cartesian.y + origin[1]),
             static_cast<type_real>(cartesian.z + origin[2]) };
  }
  const auto landing = specfem::algorithms::project_onto_surface(
      mesh, surface,
      { static_cast<type_real>(cartesian.x),
        static_cast<type_real>(cartesian.y),
        static_cast<type_real>(cartesian.z) });
  return { static_cast<type_real>(cartesian.x),
           static_cast<type_real>(cartesian.y),
           static_cast<type_real>(cartesian.z + landing.z) };
}

// Regional (Cartesian3D) mesh: geographic input is placed through the UTM
// projection, then (like any Cartesian input) against the free surface.
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::assembly::to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::cartesian3d_mesh &raw_mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;
  const specfem::coordinate_systems::utm_projection_config utm{
    raw_mesh.utm_projection_zone
  };
  const auto &surface = raw_mesh.boundaries.acoustic_free_surface;

  if (const auto *point = coordinate_conversion_impl::as<cartesian>(input))
    return coordinate_conversion_impl::resolve_cartesian(*point, mesh, surface);

  if (raw_mesh.suppress_utm_projection)
    throw std::runtime_error("specfem::assembly::to_cartesian: mesh has no "
                             "projection for non-Cartesian coordinates");
  if (const auto *geographic = coordinate_conversion_impl::as<
          specfem::coordinate_systems::geographic_coordinates>(input))
    return coordinate_conversion_impl::resolve_cartesian(
        specfem::coordinate_systems::to<cartesian>(*geographic, utm), mesh,
        surface);
  throw std::runtime_error(
      "specfem::assembly::to_cartesian: unknown coordinate type");
}

// Regional (Cartesian3D) mesh: invert the UTM projection to recover a
// geographic coordinate from a mesh-space Cartesian point.
specfem::coordinate_systems::geographic_coordinates
specfem::assembly::to_geographic(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::cartesian3d_mesh &raw_mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;
  (void)mesh;
  const specfem::coordinate_systems::utm_projection_config utm{
    raw_mesh.utm_projection_zone
  };

  if (raw_mesh.suppress_utm_projection)
    throw std::runtime_error(
        "specfem::assembly::to_geographic: mesh has no projection");

  if (const auto *point = coordinate_conversion_impl::as<cartesian>(input))
    return specfem::coordinate_systems::to<
        specfem::coordinate_systems::geographic_coordinates>(*point, utm);
  throw std::runtime_error(
      "specfem::assembly::to_geographic: expected a Cartesian input");
}
