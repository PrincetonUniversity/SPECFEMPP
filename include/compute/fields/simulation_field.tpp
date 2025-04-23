#pragma once

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
#include "compute/fields/simulation_field.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace {
template <typename ViewType> int compute_nglob(const ViewType index_mapping) {
  const int nspec = index_mapping.extent(0);
  const int ngllz = index_mapping.extent(1);
  const int ngllx = index_mapping.extent(2);

  int nglob = -1;
  // compute max value stored in index_mapping
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int igllz = 0; igllz < ngllz; igllz++) {
      for (int igllx = 0; igllx < ngllx; igllx++) {
        nglob = std::max(nglob, index_mapping(ispec, igllz, igllx));
      }
    }
  }

  return nglob + 1;
}
} // namespace

template <specfem::wavefield::simulation_field WavefieldType>
specfem::compute::simulation_field<WavefieldType>::simulation_field(
    const specfem::compute::mesh &mesh,
    const specfem::compute::element_types &element_types) {

  nglob = compute_nglob(mesh.points.h_index_mapping);

  this->nspec = mesh.points.nspec;
  this->ngllz = mesh.points.ngllz;
  this->ngllx = mesh.points.ngllx;
  this->index_mapping = mesh.points.index_mapping;
  this->h_index_mapping = mesh.points.h_index_mapping;

  assembly_index_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::DevMemSpace>(
          "specfem::compute::simulation_field::index_mapping", nglob);

  h_assembly_index_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::HostMemSpace>(
          Kokkos::create_mirror_view(assembly_index_mapping));

  for (int iglob = 0; iglob < nglob; iglob++) {
    for (int itype = 0; itype < specfem::element::ntypes; itype++) {
      h_assembly_index_mapping(iglob, itype) = -1;
    }
  }

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
                 MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH,
                  ACOUSTIC, POROELASTIC)),
      CAPTURE(field) {
        auto index = Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
                                     static_cast<int>(_medium_tag_));
        _field_ =
            specfem::compute::impl::field_impl<_dimension_tag_, _medium_tag_>(
                mesh, element_types, index);
      })

  Kokkos::deep_copy(assembly_index_mapping, h_assembly_index_mapping);

  return;
}

template <specfem::wavefield::simulation_field WavefieldType>
int specfem::compute::simulation_field<
    WavefieldType>::get_total_degrees_of_freedom() {
  if (total_degrees_of_freedom != 0) {
    return total_degrees_of_freedom;
  }

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
                 MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH,
                  ACOUSTIC, POROELASTIC)),
      CAPTURE(field) {
        total_degrees_of_freedom +=
            this->get_nglob<_medium_tag_>() *
            specfem::element::attributes<_dimension_tag_,
                                         _medium_tag_>::components;
      })

  return total_degrees_of_freedom;
} // namespace specfem::compute::simulation_field
