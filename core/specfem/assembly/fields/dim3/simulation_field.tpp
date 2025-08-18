#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields/impl/field_impl.tpp"
#include "specfem/assembly/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace {
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
} // namespace

template <specfem::wavefield::simulation_field WavefieldType>
specfem::assembly::simulation_field<specfem::dimension::type::dim3,
                                    WavefieldType>::
    simulation_field(
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types) {

  nglob = compute_nglob3D(mesh.h_index_mapping);
  this->index_mapping = mesh.index_mapping;
  this->h_index_mapping = mesh.h_index_mapping;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
      CAPTURE(assembly_index_mapping, h_assembly_index_mapping, field) {
        _assembly_index_mapping_ = Kokkos::View<int *, Kokkos::LayoutLeft,
                                                specfem::kokkos::DevMemSpace>(
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

template <specfem::wavefield::simulation_field WavefieldType>
int specfem::assembly::simulation_field<
    specfem::dimension::type::dim3,
    WavefieldType>::get_total_degrees_of_freedom() {
  if (total_degrees_of_freedom != 0) {
    return total_degrees_of_freedom;
  }

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)), CAPTURE(
                                                                      field) {
    total_degrees_of_freedom +=
        this->get_nglob<_medium_tag_>() *
        specfem::element::attributes<_dimension_tag_, _medium_tag_>::components;
  })

  return total_degrees_of_freedom;
}

template <specfem::wavefield::simulation_field WavefieldType>
template <specfem::sync::kind sync>
void specfem::assembly::simulation_field<specfem::dimension::type::dim3,
                                         WavefieldType>::sync_fields() {
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                      CAPTURE(field) { _field_.template sync_fields<sync>(); })
}
