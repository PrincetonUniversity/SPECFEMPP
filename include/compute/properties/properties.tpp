#ifndef _COMPUTE_PROPERTIES_PROPERTIES_TPP_
#define _COMPUTE_PROPERTIES_PROPERTIES_TPP_

#include "mesh/materials/interface.hpp"
#include "properties.hpp"
#include <Kokkos_Core.hpp>

template <specfem::enums::element::type type,
          specfem::enums::element::property_tag property>
specfem::compute::properties::material_property<
    type, property>::medium_property(const int nspec, const int n_element,
                                     const int ngllz, const int ngllx,
                                     const specfem::mesh::materials &materials,
                                     const specfem::kokkos::HostView1d<int>
                                         property_material_mapping)
    : specfem::compute::properties::impl::properties_container<type, property>(
          n_element, ngllz, ngllx) {

  int count = 0;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    const auto material_specification = materials.material_index_mapping(ispec);
    const int index = material_specification.index;
    if ((material_specification.type == type) &&
        (material_specification.property == property)) {
      property_index_mapping(ispec) = count;
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          // Get the material at index from mesh::materials
          auto material =
              std::get<specfem::material::material<type, property> >(
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

#endif /* _COMPUTE_PROPERTIES_PROPERTIES_TPP_ */
