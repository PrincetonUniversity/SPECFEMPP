#pragma once

#include "specfem/assembly/coordinate_conversion.hpp"
#include "specfem/setup.hpp"

#include <array>
#include <stdexcept>
#include <type_traits>

// dim2 meshes have no angular coordinate systems and no topographic depth
// resolution: the only supported target is Cartesian, and the only input is a
// Cartesian coordinate (absolute, or with an unset origin that defaults to the
// flat origin).
template <typename Target, specfem::simulation::model ModelTag>
specfem::assembly::coordinate_conversion_impl::resolved_t<Target>
specfem::assembly::to(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim2> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::mesh::mesh<ModelTag> &raw_mesh) {

  (void)mesh;
  (void)raw_mesh;

  if constexpr (std::is_same_v<
                    Target, specfem::coordinate_systems::cartesian_coordinates<
                                specfem::element::dimension_tag::dim2> >) {
    const auto *cartesian =
        coordinate_conversion_impl::as<
            specfem::coordinate_systems::cartesian_coordinates<
                specfem::element::dimension_tag::dim2> >(input);
    if (cartesian == nullptr)
      throw std::runtime_error(
          "specfem::assembly::to: a dim2 mesh resolves only Cartesian "
          "coordinates");
    const auto origin =
        cartesian->origin.value_or(std::array<double, 2>{ 0.0, 0.0 });
    return { static_cast<type_real>(cartesian->x + origin[0]),
             static_cast<type_real>(cartesian->z + origin[1]) };
  } else {
    static_assert(
        specfem::assembly::coordinate_conversion_impl::always_false_v<Target>,
        "specfem::assembly::to: a dim2 mesh supports only a Cartesian target");
  }
}
