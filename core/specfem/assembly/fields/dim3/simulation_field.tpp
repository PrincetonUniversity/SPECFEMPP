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
specfem::assembly::simulation_field<specfem::element::dimension_tag::dim3,
                                    WavefieldType>::
    simulation_field(
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types) {

  this->nspec = mesh.nspec;
  this->ngllz = mesh.element_grid.ngllz;
  this->nglly = mesh.element_grid.nglly;
  this->ngllx = mesh.element_grid.ngllx;

  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> dedup_table(
      "specfem::assembly::simulation_field::dedup_table", mesh.nglob);
  Kokkos::deep_copy(dedup_table, -1);

  this->nglob = 0;

  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    constexpr auto medium = TagsType::medium_tag;
    int nglob_M = 0;

    const auto elements = element_types.get_elements_on_host(medium);
    element_ranges.template get<TagsType>() = elements;

    auto &local = local_index_mapping.template get<TagsType>();
    local = IndexViewType(
        "specfem::assembly::simulation_field::local_index_mapping",
        elements.size(), mesh.element_grid.ngllz, mesh.element_grid.nglly,
        mesh.element_grid.ngllx);
    auto &h_local = h_local_index_mapping.template get<TagsType>();
    h_local = Kokkos::create_mirror_view(local);

    specfem::assembly::fields_impl::assign_assembly_index_mapping(
        mesh.h_index_mapping, h_local, elements, dedup_table, nglob_M);
    Kokkos::deep_copy(local, h_local);

    field.template get<TagsType>() =
        specfem::assembly::fields_impl::field_impl<dimension_tag, medium>(
            nglob_M);

    this->nglob += nglob_M;
  });

  return;
}

template <specfem::simulation::field_type WavefieldType>
int specfem::assembly::simulation_field<
    specfem::element::dimension_tag::dim3,
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
void specfem::assembly::simulation_field<specfem::element::dimension_tag::dim3,
                                         WavefieldType>::copy_to_host() {
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    field.template get<TagsType>().copy_to_host();
  });
}

template <specfem::simulation::field_type WavefieldType>
void specfem::assembly::simulation_field<specfem::element::dimension_tag::dim3,
                                         WavefieldType>::copy_to_device() {
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    field.template get<TagsType>().copy_to_device();
  });
}
