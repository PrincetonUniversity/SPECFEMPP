#pragma once

#include "assign_assembly_index_mapping.hpp"
#include "field_impl.hpp"

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::fields_impl::field_impl<DimensionTag, MediumTag>::field_impl(
    const int nglob)
    : nglob(nglob),
      displacement_base_type(nglob, "specfem::assembly::fields::displacement"),
      velocity_base_type(nglob, "specfem::assembly::fields::velocity"),
      acceleration_base_type(nglob, "specfem::assembly::fields::acceleration"),
      mass_inverse_base_type(nglob, "specfem::assembly::fields::mass_inverse") {
}

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::fields_impl::field_impl<DimensionTag, MediumTag>::field_impl(
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::element_types<dimension_tag> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
        assembly_index_mapping) {

  specfem::assembly::fields_impl::assign_assembly_index_mapping(
      mesh, element_types, assembly_index_mapping, nglob, MediumTag);

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

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::fields_impl::field_impl<DimensionTag,
                                                MediumTag>::copy_to_host() {
  displacement_base_type::copy_to_host();
  velocity_base_type::copy_to_host();
  acceleration_base_type::copy_to_host();
}

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::fields_impl::field_impl<DimensionTag,
                                                MediumTag>::copy_to_device() {
  displacement_base_type::copy_to_device();
  velocity_base_type::copy_to_device();
  acceleration_base_type::copy_to_device();
}
