#pragma once

#include "field_impl.hpp"
#include "kokkos_abstractions.h"
#include "specfem/parallel_configuration.hpp"
#include "specfem/assembly/element_types.hpp"
#include <Kokkos_Core.hpp>

namespace {
template <specfem::dimension::type DimensionTag>
void assign_assembly_index_mapping(
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::element_types<DimensionTag> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
        assembly_index_mapping,
    int &nglob, const specfem::element::medium_tag MediumTag);

template <>
void assign_assembly_index_mapping<specfem::dimension::type::dim2>(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
        assembly_index_mapping,
    int &nglob, const specfem::element::medium_tag MediumTag) {
  const auto index_mapping = mesh.h_index_mapping;
  const int nspec = mesh.nspec;
  const int ngllz = mesh.element_grid.ngllz;
  const int ngllx = mesh.element_grid.ngllx;

  int count = 0;

  constexpr int chunk_size = specfem::parallel_configuration::storage_chunk_size;
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int ix = 0; ix < ngllx; ix++) {
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          const auto medium = element_types.get_medium_tag(ispec);
          if (medium == MediumTag) {
            const int global_index =
                index_mapping(ispec, iz, ix); // get global index
            // increase the count only if the global index is not already
            // counted
            /// static_cast<int>(medium::value) is the index of the medium in
            /// the enum class
            if (assembly_index_mapping(global_index) == -1) {
              assembly_index_mapping(global_index) = count;
              count++;
            }
          }
        }
      }
    }
  }

  nglob = count;

  return;
}

template <>
void assign_assembly_index_mapping<specfem::dimension::type::dim3>(
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh,
    const specfem::assembly::element_types<specfem::dimension::type::dim3>
        &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
        assembly_index_mapping,
    int &nglob, const specfem::element::medium_tag MediumTag) {

  const auto index_mapping = mesh.h_index_mapping;
  const int nspec = mesh.nspec;
  const int ngllz = mesh.element_grid.ngllz;
  const int nglly = mesh.element_grid.nglly;
  const int ngllx = mesh.element_grid.ngllx;

  int count = 0;

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int iy = 0; iy < nglly; iy++) {
          for (int ix = 0; ix < ngllx; ix++) {
          const auto medium = element_types.get_medium_tag(ispec);
          if (medium == MediumTag) {
            const int global_index = index_mapping(ispec, iz, iy, ix);
            if (assembly_index_mapping(global_index) == -1) {
              assembly_index_mapping(global_index) = count;
              count++;
            }
          }
        }
      }
    }
  }

  nglob = count;

  return;
}

} // namespace

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::fields_impl::field_impl<DimensionTag, MediumTag>::field_impl(
    const int nglob)
    : nglob(nglob),
      displacement_base_type(nglob, "specfem::assembly::fields::displacement"),
      velocity_base_type(nglob, "specfem::assembly::fields::velocity"),
      acceleration_base_type(nglob, "specfem::assembly::fields::acceleration"),
      mass_inverse_base_type(nglob, "specfem::assembly::fields::mass_inverse") {
}

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::fields_impl::field_impl<DimensionTag, MediumTag>::field_impl(
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::element_types<dimension_tag> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
        assembly_index_mapping) {

  assign_assembly_index_mapping(mesh, element_types, assembly_index_mapping,
                                nglob, MediumTag);

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

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
template <specfem::sync::kind sync>
void specfem::assembly::fields_impl::field_impl<
    DimensionTag, MediumTag>::sync_fields() const {
  displacement_base_type::template sync<sync>();
  velocity_base_type::template sync<sync>();
  acceleration_base_type::template sync<sync>();
}
