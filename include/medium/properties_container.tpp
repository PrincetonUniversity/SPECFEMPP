#pragma once

#include "properties_container.hpp"

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::medium::properties_container<specfem::dimension::type::dim2, MediumTag, PropertyTag>::
    properties_container(
        const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const int ngllz, const int ngllx,
        const specfem::mesh::materials<dimension_tag> &materials,
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
              materials.template get_material<medium_tag, property_tag>(
                      mesh_ispec);

          // Assign the material property to the property container
          auto point_property = material.get_properties();
          this->store_host_values(
              specfem::point::index<dimension_tag>(count, iz, ix),
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

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::medium::properties_container<specfem::dimension::type::dim3, MediumTag, PropertyTag>::
    properties_container(
        const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const int ngllz, const int nglly, const int ngllx,
        const specfem::mesh::materials<dimension_tag> &materials,
        const specfem::kokkos::HostView1d<int> property_index_mapping)
    : base_type(elements.extent(0), ngllz, nglly, ngllx) {

  const int nelement = elements.extent(0);
  int count = 0;
  for (int i = 0; i < nelement; ++i) {
    const int ispec = elements(i);
    property_index_mapping(ispec) = count;
    if (medium_tag == specfem::element::medium_tag::elastic && property_tag == specfem::element::property_tag::isotropic) {
      // Handle the specific case for 3D elastic isotropic materials
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            this->h_rho(count, iz, iy, ix) = materials.rho(ispec, iz, iy, ix);
            this->h_kappa(count, iz, iy, ix) = materials.kappa(ispec, iz, iy, ix);
            this->h_mu(count, iz, iy, ix) = materials.mu(ispec, iz, iy, ix);
          }
        }
      }
    }
    count++;
  }

  this->copy_to_device();

  return;
}
