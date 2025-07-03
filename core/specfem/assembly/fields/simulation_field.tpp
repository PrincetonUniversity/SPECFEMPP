#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/element_types.hpp"
#include "field_impl.tpp"
#include "simulation_field.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
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
specfem::assembly::simulation_field<WavefieldType>::simulation_field(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::element_types<specfem::dimension::type::dim2> &element_types) {

  nglob = compute_nglob(mesh.h_index_mapping);

  this->nspec = mesh.nspec;
  this->ngllz = mesh.ngllz;
  this->ngllx = mesh.ngllx;
  this->index_mapping = mesh.index_mapping;
  this->h_index_mapping = mesh.h_index_mapping;

  assembly_index_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::DevMemSpace>(
          "specfem::assembly::simulation_field::index_mapping", nglob);

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
                  ACOUSTIC, POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(field) {
        auto index = Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
                                     static_cast<int>(_medium_tag_));
        _field_ =
            specfem::assembly::impl::field_impl<_dimension_tag_, _medium_tag_>(
                mesh, element_types, index);
      })

  Kokkos::deep_copy(assembly_index_mapping, h_assembly_index_mapping);

  return;
}

template <specfem::wavefield::simulation_field WavefieldType>
int specfem::assembly::simulation_field<
    WavefieldType>::get_total_degrees_of_freedom() {
  if (total_degrees_of_freedom != 0) {
    return total_degrees_of_freedom;
  }

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
                 MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH,
                  ACOUSTIC, POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(field) {
        total_degrees_of_freedom +=
            this->get_nglob<_medium_tag_>() *
            specfem::element::attributes<_dimension_tag_,
                                         _medium_tag_>::components;
      })

  return total_degrees_of_freedom;
}
