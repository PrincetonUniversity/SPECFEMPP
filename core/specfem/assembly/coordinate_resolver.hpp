#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/constants.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/coordinate_resolution_result.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geocentric_projection.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/transform.hpp"
#include "specfem/coordinate_systems/utm_projection.hpp"
#include "specfem/mesh.hpp"
#include "specfem/setup.hpp"
#include "specfem/simulation.hpp"

#include <memory>
#include <stdexcept>

namespace specfem {
namespace assembly {

/**
 * @brief Resolves a generic coordinate to mesh-space global coordinates.
 *
 * The base performs coordinate-type dispatch once: Cartesian input is resolved
 * by the shared @ref resolve_cartesian (absolute, or depth-based against the
 * free surface); any other (angular) coordinate is delegated to the virtual
 * @ref resolve_projected, which defaults to throwing. The base is therefore the
 * correct resolver for a plain Cartesian mesh (and for dim2, which has no
 * angular coordinates). Projection modes derive and override the hooks.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
class coordinate_resolver {
public:
  using Result =
      specfem::coordinate_systems::CoordinateResolutionResult<DimensionTag>;

  virtual ~coordinate_resolver() = default;

  /**
   * @brief Resolve a coordinate to mesh-space global coordinates.
   *
   * @param coordinate The coordinate to resolve (non-const: may set its origin)
   * @param mesh The assembled mesh (geometry context)
   * @param surface Free-surface faces for depth resolution (dim3)
   * @return Resolved global coordinates plus the found topography (if any)
   */
  Result resolve(
      specfem::coordinate_systems::coordinates<DimensionTag> &coordinate,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::mesh::acoustic_free_surface<DimensionTag> &surface) const {
    if (auto *cartesian = dynamic_cast<
            specfem::coordinate_systems::cartesian_coordinates<DimensionTag> *>(
            &coordinate))
      return resolve_cartesian(*cartesian, mesh, surface);
    return resolve_projected(coordinate, mesh, surface);
  }

protected:
  /**
   * @brief Resolve Cartesian coordinates (absolute, or depth against surface).
   *
   * Shared by every mode. Absolute coordinates (origin set) pass through;
   * depth-based coordinates (origin unset) are placed against the topographic
   * surface (dim3) or the flat fallback (dim2).
   */
  Result resolve_cartesian(
      specfem::coordinate_systems::cartesian_coordinates<DimensionTag>
          &cartesian,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::mesh::acoustic_free_surface<DimensionTag> &surface) const;

  /**
   * @brief Resolve a non-Cartesian (angular) coordinate.
   *
   * Default: the mesh has no projection configured. Projection modes override.
   */
  virtual Result resolve_projected(
      specfem::coordinate_systems::coordinates<DimensionTag> &,
      const specfem::assembly::mesh<DimensionTag> &,
      const specfem::mesh::acoustic_free_surface<DimensionTag> &) const {
    throw std::runtime_error(
        "coordinate_resolver: mesh has no projection for non-Cartesian "
        "coordinates");
  }
};

/**
 * @brief Base for dim3 projection modes: dispatches the angular coordinate
 * type.
 *
 * Centralizes the geographic / geocentric dispatch so each concrete mode only
 * implements the conversions it supports.
 */
class projecting_resolver
    : public coordinate_resolver<specfem::element::dimension_tag::dim3> {
protected:
  Result resolve_projected(
      specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3> &coordinate,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
          &mesh,
      const specfem::mesh::acoustic_free_surface<
          specfem::element::dimension_tag::dim3> &surface) const override {
    if (auto *geographic =
            dynamic_cast<specfem::coordinate_systems::geographic_coordinates *>(
                &coordinate))
      return resolve_geographic(*geographic, mesh, surface);
    if (auto *geocentric =
            dynamic_cast<specfem::coordinate_systems::geocentric_coordinates *>(
                &coordinate))
      return resolve_geocentric(*geocentric, mesh, surface);
    throw std::runtime_error("projecting_resolver: unknown coordinate type");
  }

  virtual Result resolve_geographic(
      specfem::coordinate_systems::geographic_coordinates &geographic,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
          &mesh,
      const specfem::mesh::acoustic_free_surface<
          specfem::element::dimension_tag::dim3> &surface) const = 0;

  virtual Result resolve_geocentric(
      specfem::coordinate_systems::geocentric_coordinates &,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
      const specfem::mesh::acoustic_free_surface<
          specfem::element::dimension_tag::dim3> &) const {
    throw std::runtime_error(
        "this mesh does not resolve geocentric coordinates");
  }
};

/**
 * @brief Regional mesh: geographic input is placed via the UTM projection.
 */
class utm_resolver final : public projecting_resolver {
public:
  explicit utm_resolver(
      specfem::coordinate_systems::utm_projection_config config)
      : config_(config) {}

protected:
  Result resolve_geographic(
      specfem::coordinate_systems::geographic_coordinates &geographic,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
          &mesh,
      const specfem::mesh::acoustic_free_surface<
          specfem::element::dimension_tag::dim3> &surface) const override {
    auto cartesian = specfem::coordinate_systems::transform<
        specfem::coordinate_systems::cartesian_coordinates<
            specfem::element::dimension_tag::dim3>>(geographic, config_);
    return resolve_cartesian(cartesian, mesh, surface);
  }

private:
  specfem::coordinate_systems::utm_projection_config config_;
};

/**
 * @brief Globe mesh: geographic / geocentric input is placed on the sphere.
 *
 * Perfect-sphere case only. Composes the geographic -> geocentric -> Cartesian
 * transforms; holds the planet radius and the ellipticity / topography policy
 * flags. Elliptical or topographic meshes throw rather than silently
 * approximating.
 */
class spherical_resolver final : public projecting_resolver {
public:
  spherical_resolver(double planet_radius, bool ellipticity, bool topography)
      : planet_radius_(planet_radius), ellipticity_(ellipticity),
        topography_(topography) {}

protected:
  Result resolve_geographic(
      specfem::coordinate_systems::geographic_coordinates &geographic,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
      const specfem::mesh::acoustic_free_surface<
          specfem::element::dimension_tag::dim3> &) const override {
    guard_unsupported();
    const auto geocentric = specfem::coordinate_systems::transform<
        specfem::coordinate_systems::geocentric_coordinates>(
        geographic, specfem::coordinate_systems::geocentric_projection_config{
                        reference_radius(geographic) });
    return to_result(geocentric);
  }

  Result resolve_geocentric(
      specfem::coordinate_systems::geocentric_coordinates &geocentric,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
      const specfem::mesh::acoustic_free_surface<
          specfem::element::dimension_tag::dim3> &) const override {
    guard_unsupported();
    return to_result(geocentric);
  }

private:
  // Depth is already folded into the radius (see reference_radius and the
  // geographic->geocentric transform), so there is no surface projection and no
  // topography to report -- unlike the UTM path, which resolves depth against
  // the meshed surface.
  Result to_result(const specfem::coordinate_systems::geocentric_coordinates
                       &geocentric) const {
    const auto cartesian = specfem::coordinate_systems::transform<
        specfem::coordinate_systems::cartesian_coordinates<
            specfem::element::dimension_tag::dim3>>(geocentric);
    return { { static_cast<type_real>(cartesian.x),
               static_cast<type_real>(cartesian.y),
               static_cast<type_real>(cartesian.z) },
             std::nullopt };
  }

  // The one globe-specific quantity the transform cannot know: the reference
  // surface radius along the ray. Perfect sphere returns the planet radius; the
  // elliptical/topographic case will return the deformed radius.
  double reference_radius(
      const specfem::coordinate_systems::geographic_coordinates &) const {
    return planet_radius_;
  }

  void guard_unsupported() const {
    if (ellipticity_ || topography_)
      throw std::runtime_error(
          "elliptical/topographic globe coordinate resolution not yet "
          "implemented (issue #2058)");
  }

  double planet_radius_;
  bool ellipticity_;
  bool topography_;
};

/**
 * @brief Build the coordinate resolver for a raw mesh (the one place that
 * selects the resolution mode).
 *
 * @tparam ModelTag Simulation model (selects the resolver at compile time)
 * @param raw_mesh The raw mesh carrying projection/planet metadata
 * @return A resolver owning its configuration
 */
template <specfem::simulation::model ModelTag>
auto make_coordinate_resolver(const specfem::mesh::mesh<ModelTag> &raw_mesh) {
  // dim3 resolvers return a common base pointer so each if-constexpr arm
  // deduces the same type.
  using resolver_pointer = std::unique_ptr<
      coordinate_resolver<specfem::element::dimension_tag::dim3>>;
  if constexpr (ModelTag == specfem::simulation::model::Globe3D) {
    const auto &globe = raw_mesh.globe;
    return resolver_pointer(std::make_unique<spherical_resolver>(
        globe.planet_constants.value().r_planet(),
        globe.model_config.ellipticity, globe.model_config.topography));
  } else if constexpr (ModelTag == specfem::simulation::model::Cartesian3D) {
    if (raw_mesh.suppress_utm_projection)
      return resolver_pointer(
          std::make_unique<
              coordinate_resolver<specfem::element::dimension_tag::dim3>>());
    return resolver_pointer(std::make_unique<utm_resolver>(
        specfem::coordinate_systems::utm_projection_config{
            raw_mesh.utm_projection_zone }));
  } else {
    return std::make_unique<
        coordinate_resolver<specfem::element::dimension_tag::dim2>>();
  }
}

} // namespace assembly
} // namespace specfem
