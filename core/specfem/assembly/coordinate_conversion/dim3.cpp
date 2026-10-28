#include "specfem/assembly/coordinate_conversion.hpp"

#include "specfem/algorithms/locate_point.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/coordinate_systems/basis_rotation.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/east_north_up.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geocentric_projection.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/to.hpp"
#include "specfem/coordinate_systems/utm_projection.hpp"
#include "specfem/setup.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <stdexcept>
#include <variant>

namespace specfem::assembly::coordinate_conversion_impl {
// Shared Cartesian placement helper (dim3-only, defined below). Declared here
// so the public conversions can reuse it; the free surface is read from the
// assembled mesh.
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
resolve_cartesian(
    const specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3> &cartesian,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);
} // namespace specfem::assembly::coordinate_conversion_impl

// Shared Cartesian placement: absolute coordinates pass through; a depth-based
// coordinate (origin unset) is placed against the free surface above (x, y),
// read from the assembled mesh.
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::assembly::coordinate_conversion_impl::resolve_cartesian(
    const specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3> &cartesian,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  if (cartesian.origin.has_value()) {
    const auto &origin = *cartesian.origin;
    return { static_cast<type_real>(cartesian.x + origin[0]),
             static_cast<type_real>(cartesian.y + origin[1]),
             static_cast<type_real>(cartesian.z + origin[2]) };
  }
  const auto landing = specfem::algorithms::project_onto_surface(
      mesh, mesh.free_surface,
      { static_cast<type_real>(cartesian.x),
        static_cast<type_real>(cartesian.y),
        static_cast<type_real>(cartesian.z) });
  return { static_cast<type_real>(cartesian.x),
           static_cast<type_real>(cartesian.y),
           static_cast<type_real>(cartesian.z + landing.z) };
}

// Place a coordinate into mesh space. A Cartesian input is placed directly
// against the free surface; otherwise the projection stored on the mesh selects
// the transform: pass-through (reject), regional UTM, or spherical globe.
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::assembly::to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;

  if (const auto *point = dynamic_cast<const cartesian *>(&input))
    return coordinate_conversion_impl::resolve_cartesian(*point, mesh);

  const auto &projection = mesh.projection.value;

  if (std::holds_alternative<
          specfem::assembly::mesh_impl::cartesian_projection>(projection))
    throw std::runtime_error("specfem::assembly::to_cartesian: mesh has no "
                             "projection for non-Cartesian coordinates");

  if (const auto *utm =
          std::get_if<specfem::coordinate_systems::utm_projection_config>(
              &projection)) {
    if (const auto *geographic = dynamic_cast<
            const specfem::coordinate_systems::geographic_coordinates *>(
            &input))
      return coordinate_conversion_impl::resolve_cartesian(
          specfem::coordinate_systems::to<cartesian>(*geographic, *utm), mesh);
    throw std::runtime_error(
        "specfem::assembly::to_cartesian: unknown coordinate type");
  }

  const auto &globe =
      std::get<specfem::assembly::mesh_impl::globe_projection>(projection);
  if (globe.ellipticity || globe.topography)
    throw std::runtime_error("elliptical/topographic globe coordinate "
                             "resolution not yet implemented (issue #2058)");

  if (const auto *geographic = dynamic_cast<
          const specfem::coordinate_systems::geographic_coordinates *>(
          &input)) {
    const auto geocentric = specfem::coordinate_systems::to<
        specfem::coordinate_systems::geocentric_coordinates>(
        *geographic, specfem::coordinate_systems::geocentric_projection_config{
                         globe.r_planet });
    return coordinate_conversion_impl::resolve_cartesian(
        specfem::coordinate_systems::to<cartesian>(geocentric), mesh);
  }

  if (const auto *geocentric = dynamic_cast<
          const specfem::coordinate_systems::geocentric_coordinates *>(&input))
    return coordinate_conversion_impl::resolve_cartesian(
        specfem::coordinate_systems::to<cartesian>(*geocentric), mesh);
  throw std::runtime_error(
      "specfem::assembly::to_cartesian: unknown coordinate type");
}

// Invert the placement projection to recover a geographic coordinate from a
// mesh-space Cartesian point (regional UTM inverse or globe sphere inverse).
specfem::coordinate_systems::geographic_coordinates
specfem::assembly::to_geographic(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;

  const auto &projection = mesh.projection.value;

  if (const auto *utm =
          std::get_if<specfem::coordinate_systems::utm_projection_config>(
              &projection)) {
    if (const auto *point = dynamic_cast<const cartesian *>(&input))
      return specfem::coordinate_systems::to<
          specfem::coordinate_systems::geographic_coordinates>(*point, *utm);
    throw std::runtime_error(
        "specfem::assembly::to_geographic: expected a Cartesian input");
  }

  if (const auto *globe =
          std::get_if<specfem::assembly::mesh_impl::globe_projection>(
              &projection)) {
    if (globe->ellipticity || globe->topography)
      throw std::runtime_error("elliptical/topographic globe coordinate "
                               "resolution not yet implemented (issue #2058)");
    if (const auto *point = dynamic_cast<const cartesian *>(&input)) {
      const auto geocentric = specfem::coordinate_systems::to<
          specfem::coordinate_systems::geocentric_coordinates>(*point);
      return specfem::coordinate_systems::to<
          specfem::coordinate_systems::geographic_coordinates>(
          geocentric, specfem::coordinate_systems::geocentric_projection_config{
                          globe->r_planet });
    }
    throw std::runtime_error(
        "specfem::assembly::to_geographic: expected a Cartesian input");
  }

  throw std::runtime_error(
      "specfem::assembly::to_geographic: mesh has no projection");
}

// Globe-only: recover a geocentric coordinate from a mesh-space Cartesian
// point.
specfem::coordinate_systems::geocentric_coordinates
specfem::assembly::to_geocentric(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;

  const auto *globe =
      std::get_if<specfem::assembly::mesh_impl::globe_projection>(
          &mesh.projection.value);
  if (globe == nullptr)
    throw std::runtime_error(
        "specfem::assembly::to_geocentric: mesh has no geocentric projection");
  if (globe->ellipticity || globe->topography)
    throw std::runtime_error("elliptical/topographic globe coordinate "
                             "resolution not yet implemented (issue #2058)");

  if (const auto *point = dynamic_cast<const cartesian *>(&input))
    return specfem::coordinate_systems::to<
        specfem::coordinate_systems::geocentric_coordinates>(*point);
  throw std::runtime_error(
      "specfem::assembly::to_geocentric: expected a Cartesian input");
}

// Columns e_theta = (0,-1,0), e_phi = (1,0,0), e_r = (0,0,1): the globe
// rotation at theta = 0, phi = -pi/2, written with exact entries
// (coordinate_conversion_tests asserts the equality).
specfem::assembly::LocalFrameRotation specfem::assembly::flat_mesh_rotation() {
  specfem::assembly::LocalFrameRotation rotation{};
  rotation[0][1] = 1.0;
  rotation[1][0] = -1.0;
  rotation[2][2] = 1.0;
  return rotation;
}

specfem::assembly::LocalFrameRotation specfem::assembly::local_frame_rotation(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &point,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  if (!std::holds_alternative<specfem::assembly::mesh_impl::globe_projection>(
          mesh.projection.value)) {
    return specfem::assembly::flat_mesh_rotation();
  }

  // On a globe the local frame rotates with position: recover the point's
  // colatitude/longitude from its planet-centred Cartesian position.
  const specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>
      cartesian(point.x, point.y, point.z);
  const auto geocentric = specfem::assembly::to_geocentric(cartesian, mesh);
  return specfem::coordinate_systems::rotation_matrix<
      specfem::coordinate_systems::CartesianFrame,
      specfem::coordinate_systems::GeocentricFrame>(geocentric.theta,
                                                    geocentric.phi);
}

std::array<std::array<double, 3>, 3>
specfem::assembly::local_east_north_up_basis(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &point,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  const auto rotation = specfem::assembly::local_frame_rotation(point, mesh);

  // Row i of `rotation` holds the (theta, phi, r) components of mesh axis i;
  // relabelling them to (East, North, Up) and placing them in column i gives
  // the transpose with East/North/Up as rows.
  std::array<std::array<double, 3>, 3> rows{};
  for (int axis = 0; axis < 3; ++axis) {
    const auto enu = specfem::coordinate_systems::to_east_north_up_components(
        Kokkos::Array<double, 3>{ rotation[axis][0], rotation[axis][1],
                                  rotation[axis][2] });
    rows[0][axis] = enu.east;
    rows[1][axis] = enu.north;
    rows[2][axis] = enu.up;
  }
  return rows;
}
