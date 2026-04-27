#pragma once

#include "specfem/enums.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/field_impl.tpp"
#include "specfem/assembly/mesh.hpp"
#include <Kokkos_Core.hpp>

template <specfem::simulation::field_type WavefieldType>
specfem::assembly::simulation_field<specfem::element::dimension_tag::dim2,
                                    WavefieldType>::
    simulation_field(
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types) {

  this->nglob = mesh.nglob;
  this->index_mapping = mesh.index_mapping;
  this->h_index_mapping = mesh.h_index_mapping;

  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    constexpr auto medium = TagsType::medium_tag;
    assembly_index_mapping.template get<TagsType>() =
        Kokkos::View<int *, Kokkos::LayoutLeft,
                     Kokkos::DefaultExecutionSpace::memory_space>(
            "specfem::assembly::simulation_field::index_mapping", nglob);
    h_assembly_index_mapping.template get<TagsType>() =
        Kokkos::create_mirror_view(
            assembly_index_mapping.template get<TagsType>());

    for (int iglob = 0; iglob < nglob; iglob++) {
      h_assembly_index_mapping.template get<TagsType>()(iglob) = -1;
    }

    field.template get<TagsType>() =
        specfem::assembly::fields_impl::field_impl<dimension_tag, medium>(
            mesh, element_types,
            h_assembly_index_mapping.template get<TagsType>());

    Kokkos::deep_copy(assembly_index_mapping.template get<TagsType>(),
                      h_assembly_index_mapping.template get<TagsType>());
  });

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
