#include "specfem/assembly/coordinate_conversion.hpp"

#include "specfem/assembly/coordinate_conversion/impl.hpp"
#include "specfem/setup.hpp"

#include <array>
#include <stdexcept>

// dim2 meshes have no angular coordinate systems and no topographic depth
// resolution: the only input is a Cartesian coordinate (absolute, or with an
// unset origin that defaults to the flat origin).
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
specfem::assembly::to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim2> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::mesh::cartesian2d_mesh &raw_mesh) {

  (void)mesh;
  (void)raw_mesh;

  const auto *cartesian = coordinate_conversion_impl::as<
      specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim2>>(input);
  if (cartesian == nullptr)
    throw std::runtime_error("specfem::assembly::to_cartesian: a dim2 mesh "
                             "resolves only Cartesian coordinates");
  const auto origin =
      cartesian->origin.value_or(std::array<double, 2>{ 0.0, 0.0 });
  return { static_cast<type_real>(cartesian->x + origin[0]),
           static_cast<type_real>(cartesian->z + origin[1]) };
}
