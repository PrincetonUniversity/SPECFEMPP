#pragma once

#include "specfem/enums.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/field_impl.tpp"
#include "specfem/assembly/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::simulation_fields_impl {
template <typename ViewType> int compute_nglob3D(const ViewType index_mapping) {
  const int nspec = index_mapping.extent(0);
  const int ngllz = index_mapping.extent(1);
  const int nglly = index_mapping.extent(2);
  const int ngllx = index_mapping.extent(3);

  int nglob = -1;
  // compute max value stored in index_mapping
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int igllz = 0; igllz < ngllz; igllz++) {
      for (int iglly = 0; iglly < nglly; iglly++) {
        for (int igllx = 0; igllx < ngllx; igllx++) {
          nglob = std::max(nglob, index_mapping(ispec, igllz, iglly, igllx));
        }
      }
    }
  }

  return nglob + 1;
}
} // namespace specfem::assembly::simulation_fields_impl

template <specfem::simulation::field_type WavefieldType>
specfem::assembly::simulation_field<specfem::element::dimension_tag::dim3,
                                    WavefieldType>::
    simulation_field(
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types) {

  nglob = specfem::assembly::simulation_fields_impl::compute_nglob3D(
      mesh.h_index_mapping);
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
