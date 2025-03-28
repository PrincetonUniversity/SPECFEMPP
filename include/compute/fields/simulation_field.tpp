#pragma once

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
#include "compute/fields/simulation_field.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/material_definitions.hpp"
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

#define ASSIGN_FIELDS(DIMENSION_TAG, MEDIUM_TAG)                               \
  auto CREATE_VARIABLE_NAME(index, GET_NAME(DIMENSION_TAG),                    \
                            GET_NAME(MEDIUM_TAG)) =                            \
      Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,                   \
                      static_cast<int>(GET_TAG(MEDIUM_TAG)));                  \
  this->CREATE_VARIABLE_NAME(field, GET_NAME(DIMENSION_TAG),                   \
                             GET_NAME(MEDIUM_TAG)) =                           \
      specfem::compute::impl::field_impl<GET_TAG(DIMENSION_TAG),               \
                                         GET_TAG(MEDIUM_TAG)>(                 \
          mesh, element_types,                                                 \
          CREATE_VARIABLE_NAME(index, GET_NAME(DIMENSION_TAG),                 \
                               GET_NAME(MEDIUM_TAG)));

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(ASSIGN_FIELDS, WHERE(DIMENSION_TAG_DIM2)
                                                    WHERE(MEDIUM_TAG_ELASTIC_SV,
                                                          MEDIUM_TAG_ELASTIC_SH,
                                                          MEDIUM_TAG_ACOUSTIC))

#undef ASSIGN_FIELDS

  // auto acoustic_index =
  //     Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
  //                     static_cast<int>(specfem::element::medium_tag::acoustic));

  // auto elastic_index =
  //     Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
  //                     static_cast<int>(specfem::element::medium_tag::elastic_sv));

  // elastic =
  //     specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
  //                                        specfem::element::medium_tag::elastic_sv>(
  //         mesh, element_types, elastic_index);

  // acoustic = specfem::compute::impl::field_impl<
  //     specfem::dimension::type::dim2,
  //     specfem::element::medium_tag::acoustic>( mesh, element_types,
  //     acoustic_index);

  Kokkos::deep_copy(assembly_index_mapping, h_assembly_index_mapping);

  return;
}

template <specfem::wavefield::simulation_field WavefieldType>
int specfem::compute::simulation_field<
    WavefieldType>::get_total_degrees_of_freedom() {

  if (total_degrees_of_freedom != 0) {
    return total_degrees_of_freedom;
  }

#define GET_DOF(DIMENSION_TAG, MEDIUM_TAG)                                     \
  total_degrees_of_freedom +=                                                  \
      this->get_nglob<GET_TAG(MEDIUM_TAG)>() *                                 \
      specfem::element::attributes<GET_TAG(DIMENSION_TAG),                     \
                                   GET_TAG(MEDIUM_TAG)>::components();

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(GET_DOF, WHERE(DIMENSION_TAG_DIM2)
                                               WHERE(MEDIUM_TAG_ELASTIC_SV,
                                                     MEDIUM_TAG_ELASTIC_SH,
                                                     MEDIUM_TAG_ACOUSTIC))

#undef GET_DOF

  return total_degrees_of_freedom;
}
