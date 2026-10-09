#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point/global_coordinates.hpp"
#include "specfem/simulation.hpp"

namespace specfem {
namespace assembly {

namespace coordinate_conversion_impl {

/// Always-false helper for a dependent static_assert in unsupported branches.
template <typename> inline constexpr bool always_false_v = false;

/// Recover a concrete coordinate type from the polymorphic base, or nullptr if
/// the dynamic type differs. Centralizes the one runtime step of to<>: the
/// stored coordinate's concrete type is only known at runtime.
template <typename Concrete, specfem::element::dimension_tag DimensionTag>
const Concrete *
as(const specfem::coordinate_systems::coordinates<DimensionTag> &coordinate) {
  return dynamic_cast<const Concrete *>(&coordinate);
}

/**
 * @brief Result type of @ref specfem::assembly::to for a given target system.
 *
 * A Cartesian target resolves to a mesh-space point
 * (@ref specfem::point::global_coordinates -- the type @ref
 * specfem::algorithms::locate_point consumes); every other target resolves to
 * its own coordinate type.
 */
template <typename Target> struct resolved {
  using type = Target;
};
template <specfem::element::dimension_tag DimensionTag>
struct resolved<
    specfem::coordinate_systems::cartesian_coordinates<DimensionTag>> {
  using type = specfem::point::global_coordinates<DimensionTag>;
};
template <typename Target> using resolved_t = typename resolved<Target>::type;

/**
 * @brief Place a dim3 Cartesian coordinate in mesh space.
 *
 * Absolute coordinates (origin set) pass through; depth-based coordinates
 * (origin unset) are placed against the topographic surface above (x, y), or
 * the flat fallback (z = 0) when the mesh has no free surface. Shared by the
 * regional and globe conversion paths.
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
resolve_cartesian(
    const specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3> &cartesian,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::acoustic_free_surface<
        specfem::element::dimension_tag::dim3> &surface);

/**
 * @brief dim3 conversion for a regional (Cartesian3D) mesh: geographic input is
 * placed via the UTM projection; Cartesian input is placed directly.
 */
template <typename Target>
resolved_t<Target> convert_regional(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::cartesian3d_mesh &raw_mesh);

/**
 * @brief dim3 conversion for a globe (Globe3D) mesh: composes the geographic /
 * geocentric / Cartesian transforms on the sphere (perfect-sphere case only).
 */
template <typename Target>
resolved_t<Target> convert_globe(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::globe3d_mesh &raw_mesh);

} // namespace coordinate_conversion_impl

/**
 * @brief Convert an input coordinate to the @p Target system using the mesh.
 *
 * Single entry point for coordinate conversion at assembly time. The projection
 * is chosen statically from @p ModelTag (UTM for a regional Cartesian3D mesh,
 * the geographic -> geocentric -> Cartesian composition for a Globe3D mesh,
 * pass-through for a plain Cartesian mesh), so callers never pass a config. The
 * concrete source type is recovered from the polymorphic @p input at runtime.
 *
 * Supported today: @c cartesian_coordinates target from any input (the
 * source/receiver placement path, returning @ref
 * specfem::point::global_coordinates); and @c geocentric_coordinates /
 * @c geographic_coordinates targets from a Cartesian input.
 *
 * @code
 * auto gcoord = specfem::assembly::to<
 *     specfem::coordinate_systems::cartesian_coordinates<dim3> >(
 *     read_coords, mesh, raw_mesh);
 * @endcode
 *
 * @tparam Target Target coordinate system (a @ref specfem::coordinate_systems
 * type)
 * @tparam ModelTag Simulation model (deduced from @p raw_mesh)
 * @param input Input coordinate to convert
 * @param mesh Assembled mesh geometry (used to place a depth against the
 * surface)
 * @param raw_mesh Raw mesh carrying the projection config and free-surface
 * faces
 * @return The resolved coordinate: @ref specfem::point::global_coordinates for
 * a Cartesian target, otherwise the target coordinate type
 */
template <typename Target, specfem::simulation::model ModelTag>
coordinate_conversion_impl::resolved_t<Target>
to(const specfem::coordinate_systems::coordinates<
       specfem::element::dimension_tag::dim2> &input,
   const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
   const specfem::mesh::mesh<ModelTag> &raw_mesh);

/// @copydoc to
template <typename Target, specfem::simulation::model ModelTag>
coordinate_conversion_impl::resolved_t<Target>
to(const specfem::coordinate_systems::coordinates<
       specfem::element::dimension_tag::dim3> &input,
   const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
   const specfem::mesh::mesh<ModelTag> &raw_mesh);

} // namespace assembly
} // namespace specfem

// Explicit instantiations live in coordinate_conversion.cpp (one per supported
// target/model), so the heavy transform / projection / project_onto_surface
// includes stay out of the call sites.

// Cartesian target (source/receiver placement) -- one per model.
extern template specfem::point::global_coordinates<
    specfem::element::dimension_tag::dim2>
specfem::assembly::to<specfem::coordinate_systems::cartesian_coordinates<
                          specfem::element::dimension_tag::dim2>,
                      specfem::simulation::model::Cartesian2D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim2> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &,
    const specfem::mesh::mesh<specfem::simulation::model::Cartesian2D> &);

extern template specfem::point::global_coordinates<
    specfem::element::dimension_tag::dim3>
specfem::assembly::to<specfem::coordinate_systems::cartesian_coordinates<
                          specfem::element::dimension_tag::dim3>,
                      specfem::simulation::model::Cartesian3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Cartesian3D> &);

extern template specfem::point::global_coordinates<
    specfem::element::dimension_tag::dim3>
specfem::assembly::to<specfem::coordinate_systems::cartesian_coordinates<
                          specfem::element::dimension_tag::dim3>,
                      specfem::simulation::model::Globe3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Globe3D> &);

// Cartesian -> geocentric (globe).
extern template specfem::coordinate_systems::geocentric_coordinates
specfem::assembly::to<specfem::coordinate_systems::geocentric_coordinates,
                      specfem::simulation::model::Globe3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Globe3D> &);

// Cartesian -> geographic (regional UTM, and globe).
extern template specfem::coordinate_systems::geographic_coordinates
specfem::assembly::to<specfem::coordinate_systems::geographic_coordinates,
                      specfem::simulation::model::Cartesian3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Cartesian3D> &);

extern template specfem::coordinate_systems::geographic_coordinates
specfem::assembly::to<specfem::coordinate_systems::geographic_coordinates,
                      specfem::simulation::model::Globe3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Globe3D> &);
