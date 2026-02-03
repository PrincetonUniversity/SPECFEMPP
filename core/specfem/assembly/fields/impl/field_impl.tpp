#pragma once

#include "field_impl.hpp"
#include "assign_assembly_index_mapping.hpp"


template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::fields_impl::field_impl<DimensionTag, MediumTag>::field_impl(
    const int nglob)
    : nglob(nglob),
      displacement_base_type(nglob, "specfem::assembly::fields::displacement"),
      velocity_base_type(nglob, "specfem::assembly::fields::velocity"),
      acceleration_base_type(nglob, "specfem::assembly::fields::acceleration"),
      mass_inverse_base_type(nglob, "specfem::assembly::fields::mass_inverse") {
}

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::fields_impl::field_impl<DimensionTag, MediumTag>::field_impl(
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::element_types<dimension_tag> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
        assembly_index_mapping) {

  specfem::assembly::fields_impl::assign_assembly_index_mapping(mesh, element_types, assembly_index_mapping,
                                nglob, MediumTag);

  static_cast<displacement_base_type &>(*this) =
      displacement_base_type(nglob, "specfem::assembly::fields::displacement");
  static_cast<velocity_base_type &>(*this) =
      velocity_base_type(nglob, "specfem::assembly::fields::velocity");
  static_cast<acceleration_base_type &>(*this) =
      acceleration_base_type(nglob, "specfem::assembly::fields::acceleration");
  static_cast<mass_inverse_base_type &>(*this) =
      mass_inverse_base_type(nglob, "specfem::assembly::fields::mass_inverse");

  return;
}

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
template <specfem::sync::kind sync>
void specfem::assembly::fields_impl::field_impl<
    DimensionTag, MediumTag>::sync_fields() const {
  displacement_base_type::template sync<sync>();
  velocity_base_type::template sync<sync>();
  acceleration_base_type::template sync<sync>();
}
