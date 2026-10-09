#pragma once

#include "specfem/algorithms/locate_point.hpp"
#include "specfem/assembly/coordinate_conversion.hpp"
#include "specfem/assembly/coordinate_conversion/dim3/globe.tpp"
#include "specfem/assembly/coordinate_conversion/dim3/regional.tpp"
#include "specfem/setup.hpp"

#include <type_traits>

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

// dim3 dispatcher: the regional and globe paths are completely separate, so
// select one statically from the mesh model.
template <typename Target, specfem::simulation::model ModelTag>
specfem::assembly::coordinate_conversion_impl::resolved_t<Target>
specfem::assembly::to(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::mesh<ModelTag> &raw_mesh) {

  if constexpr (ModelTag == specfem::simulation::model::Cartesian3D) {
    return specfem::assembly::coordinate_conversion_impl::convert_regional<
        Target>(input, mesh, raw_mesh);
  } else if constexpr (ModelTag == specfem::simulation::model::Globe3D) {
    return specfem::assembly::coordinate_conversion_impl::convert_globe<Target>(
        input, mesh, raw_mesh);
  } else {
    static_assert(
        specfem::assembly::coordinate_conversion_impl::always_false_v<Target>,
        "specfem::assembly::to: unsupported dim3 simulation model");
  }
}
