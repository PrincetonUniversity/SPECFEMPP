#pragma once

#include "domain_properties.hpp"

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::assembly::impl::domain_properties<
    specfem::element::dimension_tag::dim2, MediumTag, PropertyTag>::
    domain_properties(const specfem::datatype::ElementIndexRange &elements,
                      const specfem::assembly::mesh<dimension_tag> &mesh,
                      const specfem::mesh::materials<dimension_tag> &materials,
                      const bool has_gll_model)
    : base_type(elements.extent(0), mesh.element_grid) {

  const int nelement = elements.extent(0);
  element_range = elements;
  for (int i = 0; i < nelement; ++i) {
    const int ispec = elements(i);
    const int mesh_ispec = mesh.h_compute_to_mesh(ispec);
    const int count = i;
    if (!has_gll_model) {
      for (int iz = 0; iz < mesh.element_grid.ngllz; ++iz) {
        for (int ix = 0; ix < mesh.element_grid.ngllx; ++ix) {
          auto point_property =
              materials.template get_properties<medium_tag, property_tag>(
                  mesh_ispec);
          this->store_host_values(
              specfem::point::index<dimension_tag>(count, iz, ix),
              point_property);
        }
      }
    }
  }

  if (!has_gll_model) {
    this->copy_to_device();
  }

  return;
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::assembly::impl::domain_properties<
    specfem::element::dimension_tag::dim3, MediumTag, PropertyTag>::
    domain_properties(const specfem::datatype::ElementIndexRange &elements,
                      const specfem::assembly::mesh<dimension_tag> &mesh,
                      const specfem::mesh::materials<dimension_tag> &materials,
                      const bool has_gll_model)
    : base_type(elements.extent(0), mesh.element_grid) {

  const int nelement = elements.extent(0);
  element_range = elements;
  Kokkos::parallel_for(
      "specfem::assembly::impl::domain_properties::dim3::init_host_values",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<4>>(
          { 0, 0, 0, 0 }, { nelement, mesh.element_grid.ngllz,
                            mesh.element_grid.nglly, mesh.element_grid.ngllx }),
      [=, this](const int i, const int iz, const int iy, const int ix) {
        const int ispec = elements(i);
        const int mesh_ispec = mesh.h_compute_to_mesh(ispec);
        if (!has_gll_model) {
          const auto point_property =
              materials.template get_properties<medium_tag, property_tag>(
                  mesh_ispec);
          this->store_host_values(
              specfem::point::index<dimension_tag, false>(i, iz, iy, ix),
              point_property);
        }
      });

  if (!has_gll_model) {
    this->copy_to_device();
  }
  return;
}
