#ifndef _EVENT_MARCHING_BUILD_DG_FIELDS_TPP
#define _EVENT_MARCHING_BUILD_DG_FIELDS_TPP



void rebuild_simulation_field(){

}





template <specfem::wavefield::type WavefieldType>
specfem::compute::simulation_field<WavefieldType>::simulation_field(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties) {

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

  auto acoustic_index =
      Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
                      static_cast<int>(acoustic_type::medium_tag));

  auto elastic_index =
      Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
                      static_cast<int>(elastic_type::medium_tag));

  elastic =
      specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                         specfem::element::medium_tag::elastic>(
          mesh, properties, elastic_index);

  acoustic = specfem::compute::impl::field_impl<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>(
      mesh, properties, acoustic_index);

  Kokkos::deep_copy(assembly_index_mapping, h_assembly_index_mapping);

  return;
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
specfem::compute::impl::field_impl<DimensionType, MediumTag>::field_impl(
    const int nglob)
    : nglob(nglob),
      field("specfem::compute::fields::field", nglob, medium_type::components),
      h_field(Kokkos::create_mirror_view(field)),
      field_dot("specfem::compute::fields::field_dot", nglob,
                medium_type::components),
      h_field_dot(Kokkos::create_mirror_view(field_dot)),
      field_dot_dot("specfem::compute::fields::field_dot_dot", nglob,
                    medium_type::components),
      h_field_dot_dot(Kokkos::create_mirror_view(field_dot_dot)),
      mass_inverse("specfem::compute::fields::mass_inverse", nglob,
                   medium_type::components),
      h_mass_inverse(Kokkos::create_mirror_view(mass_inverse)) {}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
specfem::compute::impl::field_impl<DimensionType, MediumTag>::field_impl(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties,
    Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
        assembly_index_mapping) {

  const auto index_mapping = mesh.points.h_index_mapping;
  const auto element_type = properties.h_element_types;
  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  // Count the total number of distinct global indices for the medium
  int count = 0;

  for (int ix = 0; ix < ngllx; ++ix) {
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int ispec = 0; ispec < nspec; ++ispec) {
        if (element_type(ispec) == MediumTag) {
          const int index = index_mapping(ispec, iz, ix); // get global index
          // increase the count only if the global index is not already counted
          /// static_cast<int>(medium::value) is the index of the medium in the
          /// enum class
          if (assembly_index_mapping(index) == -1) {
            assembly_index_mapping(index) = count;
            count++;
          }
        }
      }
    }
  }

  nglob = count;

  field = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field", nglob, medium_type::components);
  h_field = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(field));
  field_dot = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field_dot", nglob, medium_type::components);
  h_field_dot = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(field_dot));
  field_dot_dot = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field_dot_dot", nglob,
      medium_type::components);
  h_field_dot_dot =
      specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
          Kokkos::create_mirror_view(field_dot_dot));
  mass_inverse = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::mass_inverse", nglob, medium_type::components);
  h_mass_inverse = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(mass_inverse));

  Kokkos::parallel_for(
      "specfem::compute::fields::field_impl::initialize_field",
      specfem::kokkos::HostRange(0, nglob), [=](const int &iglob) {
        for (int icomp = 0; icomp < medium_type::components; ++icomp) {
          h_field(iglob, icomp) = 0.0;
          h_field_dot(iglob, icomp) = 0.0;
          h_field_dot_dot(iglob, icomp) = 0.0;
          h_mass_inverse(iglob, icomp) = 0.0;
        }
      });

  Kokkos::fence();

  Kokkos::deep_copy(field, h_field);
  Kokkos::deep_copy(field_dot, h_field_dot);
  Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  Kokkos::deep_copy(mass_inverse, h_mass_inverse);

  return;
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
template <specfem::sync::kind sync>
void specfem::compute::impl::field_impl<DimensionType, MediumTag>::sync_fields()
    const {
  if constexpr (sync == specfem::sync::kind::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
    Kokkos::deep_copy(h_field_dot, field_dot);
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if constexpr (sync == specfem::sync::kind::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
    Kokkos::deep_copy(field_dot, h_field_dot);
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  }
}

#endif
