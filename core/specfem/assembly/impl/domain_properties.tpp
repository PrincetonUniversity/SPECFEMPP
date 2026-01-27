#pragma once

#include "domain_properties.hpp"

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::medium::domain_properties<specfem::dimension::type::dim2, MediumTag,
                                      PropertyTag>::
    domain_properties(
        const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
        const specfem::assembly::mesh<dimension_tag> &mesh, const int ngllz,
        const int ngllx,
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
specfem::medium::domain_properties<specfem::dimension::type::dim3, MediumTag,
                                      PropertyTag>::
    domain_properties(
        const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
        const int nspec, const int ngllz, const int nglly, const int ngllx,
        const specfem::mesh::materials<dimension_tag> &materials,
        const specfem::kokkos::HostView1d<int> property_index_mapping)
    : base_type(elements.extent(0), ngllz, nglly, ngllx) {

  const int nelement = elements.extent(0);
  int count = 0;
  Kokkos::parallel_for(
      "specfem::medium::domain_properties::dim3::init_host_values",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<4> >(
          { 0, 0, 0, 0 }, { nelement, ngllz, nglly, ngllx }),
      [=](const int i, const int iz, const int iy, const int ix) {
        const int ispec = elements(i);
        property_index_mapping(ispec) = count;
        if (medium_tag == specfem::element::medium_tag::elastic &&
            property_tag == specfem::element::property_tag::isotropic) {
          // Handle the specific case
          const auto point_property =
              materials.template get_properties<medium_tag, property_tag>(
                  ispec);
          this->store_host_values(
              specfem::point::index<dimension_tag, false>(count, iz, iy, ix),
              point_property);
        }
      });

  this->copy_to_device();
  return;
}
