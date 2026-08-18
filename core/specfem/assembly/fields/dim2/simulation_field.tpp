#pragma once

#include "specfem/enums.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/assign_assembly_index_mapping.hpp"
#include "specfem/assembly/fields/impl/field_impl.tpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include <Kokkos_Core.hpp>

template <specfem::simulation::field_type WavefieldType>
specfem::assembly::simulation_field<specfem::element::dimension_tag::dim2,
                                    WavefieldType>::
    simulation_field(
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types) {

  this->nspec = mesh.nspec;
  this->ngllz = mesh.element_grid.ngllz;
  this->ngllx = mesh.element_grid.ngllx;

  // Private copy of h_index_mapping so multiple simulation_fields constructed
  // from the same mesh don't interfere when we remap in-place below.
  this->h_index_mapping = IndexViewType::host_mirror_type(
      "specfem::assembly::simulation_field::h_index_mapping",
      mesh.nspec, mesh.element_grid.ngllz, mesh.element_grid.ngllx);
  Kokkos::deep_copy(this->h_index_mapping, mesh.h_index_mapping);

  this->index_mapping = IndexViewType(
      "specfem::assembly::simulation_field::index_mapping",
      mesh.nspec, mesh.element_grid.ngllz, mesh.element_grid.ngllx);

  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> dedup_table(
      "specfem::assembly::simulation_field::dedup_table", mesh.nglob);
  Kokkos::deep_copy(dedup_table, -1);

  int base_dof = 0;

  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    constexpr auto medium = TagsType::medium_tag;
    int nglob_M = 0;

    specfem::assembly::fields_impl::assign_assembly_index_mapping(
        this->h_index_mapping, element_types, dedup_table, nglob_M, medium,
        base_dof);

    dof_ranges.template get<TagsType>() =
        specfem::datatype::ElementIndexRange(base_dof, base_dof + nglob_M);

    field.template get<TagsType>() =
        specfem::assembly::fields_impl::field_impl<dimension_tag, medium>(
            nglob_M);

    base_dof += nglob_M;
  });

  this->nglob = base_dof;
  Kokkos::deep_copy(this->index_mapping, this->h_index_mapping);

  return;
}

template <specfem::simulation::field_type WavefieldType>
int specfem::assembly::simulation_field<
    specfem::element::dimension_tag::dim2,
    WavefieldType>::get_total_degrees_of_freedom() {
  if (total_degrees_of_freedom != 0) {
    return total_degrees_of_freedom;
  }

  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    constexpr auto medium = TagsType::medium_tag;
    total_degrees_of_freedom +=
        this->get_nglob<medium>() *
        specfem::element::attributes<dimension_tag, medium>::components;
  });

  return total_degrees_of_freedom;
}

template <specfem::simulation::field_type WavefieldType>
void specfem::assembly::simulation_field<specfem::element::dimension_tag::dim2,
                                         WavefieldType>::copy_to_host() {
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    field.template get<TagsType>().copy_to_host();
  });
}

template <specfem::simulation::field_type WavefieldType>
void specfem::assembly::simulation_field<specfem::element::dimension_tag::dim2,
                                         WavefieldType>::copy_to_device() {
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    field.template get<TagsType>().copy_to_device();
  });
}
