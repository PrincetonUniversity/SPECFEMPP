#pragma once

#include "properties_container.hpp"

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::medium::properties_container<specfem::dimension::type::dim2, MediumTag, PropertyTag>::
    properties_container(
        const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
        const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
        const int ngllz, const int ngllx,
        const specfem::mesh::materials<specfem::dimension::type::dim2>
            &materials,
        const bool has_gll_model,
        const specfem::kokkos::HostView1d<int> property_index_mapping)
    : base_type(elements.extent(0), ngllz, ngllx) {

  const int nelement = elements.extent(0);
  int count = 0;
  for (int i = 0; i < nelement; ++i) {
    const int ispec = elements(i);
    const int mesh_ispec = mesh.compute_to_mesh(ispec);
    property_index_mapping(ispec) = count;
    if (!has_gll_model) {
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          // Get the material at index from mesh::materials
          auto material =
              materials
                  .get_material<base_type::medium_tag, base_type::property_tag>(
                      mesh_ispec);

          // Assign the material property to the property container
          auto point_property = material.get_properties();
          this->store_host_values(
              specfem::point::index<base_type::dimension_tag>(count, iz, ix),
              point_property);
        }
      }
    }
    count++;
  }

  if (!has_gll_model) {
    this->copy_to_device();
  }

  return;
}
