#pragma once

#include "properties.hpp"
#include <Kokkos_Core.hpp>

template <specfem::enums::element::type type,
          specfem::enums::element::property_tag property>
specfem::medium::material_properties<
    type, property>::medium_property(const int nspec, const int n_element,
                                     const int ngllz, const int ngllx,
                                     const specfem::mesh::materials &materials,
                                     const specfem::kokkos::HostView1d<int>
                                         property_material_mapping)
    : specfem::medium::properties_container<type, property>(
          n_element, ngllz, ngllx) {

  int count = 0;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    const auto material_specification = materials.material_index_mapping(ispec);
    const int index = material_specification.index;
    if ((material_specification.type == type) &&
        (material_specification.property == property)) {
      h_property_index_mapping(ispec) = count;
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          // Get the material at index from mesh::materials
          auto material =
              std::get<specfem::medium::material<type, property> >(
                  materials[index]);
          // Assign the material property to the property container
          auto point_property = material.get_property();
          this->assign(count, iz, ix, property);
        }
      }
      count++;
    }
  }

  assert(count == n_element);

  this->copy_to_device();

  return;
}
