#pragma once

#include "enumerations/medium.hpp"

namespace specfem {
namespace medium {

namespace impl {
template <typename PropertiesContainer>
void constructor(
    const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
    const int ngllz, const int ngllx, const specfem::mesh::materials &materials,
    const bool has_gll_model,
    const specfem::kokkos::HostView1d<int> property_index_mapping,
    PropertiesContainer &properties) {

  constexpr auto dimension = PropertiesContainer::dimension;
  constexpr auto type = PropertiesContainer::medium_tag;
  constexpr auto property = PropertiesContainer::property_type;

  const int nelement = elements.extent(0);
  int count = 0;
  for (int i = 0; i < nelement; ++i) {
    const int ispec = elements(i);
    property_index_mapping(ispec) = count;
    if (!has_gll_model) {
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          // Get the material at index from mesh::materials
          auto material = std::get<specfem::medium::material<type, property> >(
              materials[ispec]);

          // Assign the material property to the property container
          auto point_property = material.get_properties();
          properties.assign(specfem::point::index<dimension>(count, iz, ix),
                            point_property);
        }
      }
    }
    count++;
  }

  if (!has_gll_model) {
    properties.copy_to_device();
  }

  return;
}

} // namespace impl

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
struct properties_container {
  static_assert("Material type not implemented");
};

} // namespace medium
} // namespace specfem

// Including the template specializations here so that properties_container is
// an interface to the compute module
#include "dim2/acoustic/isotropic/properties_container.hpp"
#include "dim2/elastic/anisotropic/properties_container.hpp"
#include "dim2/elastic/isotropic/properties_container.hpp"
