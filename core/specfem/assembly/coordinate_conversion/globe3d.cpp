#include "specfem/assembly/coordinate_conversion.hpp"

#include "specfem/assembly/coordinate_conversion/impl.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geocentric_projection.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/to.hpp"
#include "specfem/globe/planet_constants.hpp"
#include "specfem/mesh.hpp"
#include "specfem/setup.hpp"

#include <optional>
#include <stdexcept>

// Globe (Globe3D) meshes compose the geographic -> geocentric -> Cartesian
// transforms on the sphere. Perfect-sphere case only; elliptical / topographic
// meshes throw rather than silently approximating (issue #2058).

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::assembly::to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::globe3d_mesh &raw_mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;

  const auto &globe = raw_mesh.globe;
  const auto &surface = raw_mesh.boundaries.acoustic_free_surface;

  if (const auto *point = coordinate_conversion_impl::as<cartesian>(input))
    return coordinate_conversion_impl::resolve_cartesian(*point, mesh, surface);

  if (globe.model_config.ellipticity || globe.model_config.topography)
    throw std::runtime_error("elliptical/topographic globe coordinate "
                             "resolution not yet implemented (issue #2058)");

  const double r_planet = globe.planet_constants.value().r_planet();

  if (const auto *geographic = coordinate_conversion_impl::as<
          specfem::coordinate_systems::geographic_coordinates>(input)) {
    const auto geocentric = specfem::coordinate_systems::to<
        specfem::coordinate_systems::geocentric_coordinates>(
        *geographic,
        specfem::coordinate_systems::geocentric_projection_config{ r_planet });
    return coordinate_conversion_impl::resolve_cartesian(
        specfem::coordinate_systems::to<cartesian>(geocentric), mesh, surface);
  }

  if (const auto *geocentric = coordinate_conversion_impl::as<
          specfem::coordinate_systems::geocentric_coordinates>(input))
    return coordinate_conversion_impl::resolve_cartesian(
        specfem::coordinate_systems::to<cartesian>(*geocentric), mesh, surface);
  throw std::runtime_error(
      "specfem::assembly::to_cartesian: unknown coordinate type");
}

specfem::coordinate_systems::geocentric_coordinates
specfem::assembly::to_geocentric(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::globe3d_mesh &raw_mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;
  (void)mesh;

  const auto &globe = raw_mesh.globe;
  if (globe.model_config.ellipticity || globe.model_config.topography)
    throw std::runtime_error("elliptical/topographic globe coordinate "
                             "resolution not yet implemented (issue #2058)");

  if (const auto *point = coordinate_conversion_impl::as<cartesian>(input))
    return specfem::coordinate_systems::to<
        specfem::coordinate_systems::geocentric_coordinates>(*point);
  throw std::runtime_error(
      "specfem::assembly::to_geocentric: expected a Cartesian input");
}

specfem::coordinate_systems::geographic_coordinates
specfem::assembly::to_geographic(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::globe3d_mesh &raw_mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;
  (void)mesh;

  const auto &globe = raw_mesh.globe;
  if (globe.model_config.ellipticity || globe.model_config.topography)
    throw std::runtime_error("elliptical/topographic globe coordinate "
                             "resolution not yet implemented (issue #2058)");
  const double r_planet = globe.planet_constants.value().r_planet();

  if (const auto *point = coordinate_conversion_impl::as<cartesian>(input)) {
    const auto geocentric = specfem::coordinate_systems::to<
        specfem::coordinate_systems::geocentric_coordinates>(*point);
    return specfem::coordinate_systems::to<
        specfem::coordinate_systems::geographic_coordinates>(
        geocentric,
        specfem::coordinate_systems::geocentric_projection_config{ r_planet });
  }
  throw std::runtime_error(
      "specfem::assembly::to_geographic: expected a Cartesian input");
}
