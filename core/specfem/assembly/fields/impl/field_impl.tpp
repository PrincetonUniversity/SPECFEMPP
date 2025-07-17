#pragma once

#include "field_impl.hpp"
#include "kokkos_abstractions.h"
#include "parallel_configuration/chunk_config.hpp"
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
  const int ngllz = mesh.ngllz;
  const int ngllx = mesh.ngllx;

  int count = 0;

  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;
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
  const int ngllz = mesh.ngllz;
  const int nglly = mesh.nglly;
  const int ngllx = mesh.ngllx;

  int count = 0;

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int ix = 0; ix < ngllx; ix++) {
      for (int iy = 0; iy < nglly; iy++) {
        for (int iz = 0; iz < ngllz; iz++) {
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
      specfem::assembly::fields_impl::field<DimensionTag, MediumTag>(nglob),
      specfem::assembly::fields_impl::field_dot<DimensionTag, MediumTag>(nglob),
      specfem::assembly::fields_impl::field_dot_dot<DimensionTag, MediumTag>(
          nglob),
      specfem::assembly::fields_impl::mass_inverse<DimensionTag, MediumTag>(
          nglob) {}

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::fields_impl::field_impl<DimensionTag, MediumTag>::field_impl(
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::element_types<dimension_tag> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
        assembly_index_mapping) {

  assign_assembly_index_mapping(mesh, element_types, assembly_index_mapping,
                                nglob, MediumTag);

  static_cast<specfem::assembly::fields_impl::field<DimensionTag, MediumTag> &>(
      *this) =
      specfem::assembly::fields_impl::field<DimensionTag, MediumTag>(nglob);
  static_cast<
      specfem::assembly::fields_impl::field_dot<DimensionTag, MediumTag> &>(
      *this) =
      specfem::assembly::fields_impl::field_dot<DimensionTag, MediumTag>(nglob);
  static_cast<
      specfem::assembly::fields_impl::field_dot_dot<DimensionTag, MediumTag> &>(
      *this) =
      specfem::assembly::fields_impl::field_dot_dot<DimensionTag, MediumTag>(
          nglob);
  static_cast<
      specfem::assembly::fields_impl::mass_inverse<DimensionTag, MediumTag> &>(
      *this) =
      specfem::assembly::fields_impl::mass_inverse<DimensionTag, MediumTag>(
          nglob);

  // field = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
  //     "specfem::assembly::fields::field", nglob, components);
  // h_field = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
  //     Kokkos::create_mirror_view(field));
  // field_dot = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
  //     "specfem::assembly::fields::field_dot", nglob, components);
  // h_field_dot = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
  //     Kokkos::create_mirror_view(field_dot));
  // field_dot_dot = specfem::kokkos::DeviceView2d<type_real,
  // Kokkos::LayoutLeft>(
  //     "specfem::assembly::fields::field_dot_dot", nglob, components);
  // h_field_dot_dot =
  //     specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
  //         Kokkos::create_mirror_view(field_dot_dot));
  // mass_inverse = specfem::kokkos::DeviceView2d<type_real,
  // Kokkos::LayoutLeft>(
  //     "specfem::assembly::fields::mass_inverse", nglob, components);
  // h_mass_inverse = specfem::kokkos::HostMirror2d<type_real,
  // Kokkos::LayoutLeft>(
  //     Kokkos::create_mirror_view(mass_inverse));

  // Kokkos::parallel_for(
  //     "specfem::assembly::fields::field_impl::initialize_field",
  //     specfem::kokkos::HostRange(0, nglob), [=](const int &iglob) {
  //       for (int icomp = 0; icomp < components; ++icomp) {
  //         h_field(iglob, icomp) = 0.0;
  //         h_field_dot(iglob, icomp) = 0.0;
  //         h_field_dot_dot(iglob, icomp) = 0.0;
  //         h_mass_inverse(iglob, icomp) = 0.0;
  //       }
  //     });

  // Kokkos::fence();

  // Kokkos::deep_copy(field, h_field);
  // Kokkos::deep_copy(field_dot, h_field_dot);
  // Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  // Kokkos::deep_copy(mass_inverse, h_mass_inverse);

  return;
}

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
template <specfem::sync::kind sync>
void specfem::assembly::fields_impl::field_impl<
    DimensionTag, MediumTag>::sync_fields() const {
  static_cast<
      const specfem::assembly::fields_impl::field<DimensionTag, MediumTag> &>(
      *this)
      .template sync<sync>();
  static_cast<const specfem::assembly::fields_impl::field_dot<DimensionTag,
                                                             MediumTag> &>(
      *this)
      .template sync<sync>();
  static_cast<const specfem::assembly::fields_impl::field_dot_dot<DimensionTag,
                                                                 MediumTag> &>(
      *this)
      .template sync<sync>();
}
