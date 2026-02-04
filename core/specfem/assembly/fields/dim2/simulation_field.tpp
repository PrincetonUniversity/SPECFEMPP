#pragma once

#include "enumerations/interface.hpp"

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/field_impl.tpp"
#include "specfem/assembly/mesh.hpp"
#include <Kokkos_Core.hpp>

template <specfem::simulation::field_type WavefieldType>
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    WavefieldType>::
    simulation_field(
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types) {

  this->nglob = mesh.nglob;
  this->index_mapping = mesh.index_mapping;
  this->h_index_mapping = mesh.h_index_mapping;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(assembly_index_mapping, h_assembly_index_mapping, field) {
        _assembly_index_mapping_ =
            Kokkos::View<int *, Kokkos::LayoutLeft,
                         Kokkos::DefaultExecutionSpace::memory_space>(
                "specfem::assembly::simulation_field::index_mapping", nglob);
        _h_assembly_index_mapping_ =
            Kokkos::create_mirror_view(_assembly_index_mapping_);

        for (int iglob = 0; iglob < nglob; iglob++) {
          _h_assembly_index_mapping_(iglob) = -1;
        }

        _field_ = specfem::assembly::fields_impl::field_impl<_dimension_tag_,
                                                             _medium_tag_>(
            mesh, element_types, _h_assembly_index_mapping_);

        Kokkos::deep_copy(_assembly_index_mapping_, _h_assembly_index_mapping_);
      })

  return;
}

template <specfem::simulation::field_type WavefieldType>
int specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    WavefieldType>::get_total_degrees_of_freedom() {
  if (total_degrees_of_freedom != 0) {
    return total_degrees_of_freedom;
  }

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(field) {
        total_degrees_of_freedom +=
            this->get_nglob<_medium_tag_>() *
            specfem::element::attributes<_dimension_tag_,
                                         _medium_tag_>::components;
      })

  return total_degrees_of_freedom;
}

template <specfem::simulation::field_type WavefieldType>
void specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                         WavefieldType>::copy_to_host() {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(field) { _field_.copy_to_host(); })
}

template <specfem::simulation::field_type WavefieldType>
void specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                         WavefieldType>::copy_to_device() {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(field) { _field_.copy_to_device(); })
}
