#include "compute/element_types/element_types.hpp"

specfem::compute::element_types::element_types(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags)
    : nspec(nspec),
      medium_tags("specfem::compute::element_types::medium_tags", nspec),
      property_tags("specfem::compute::element_types::property_tags", nspec),
      boundary_tags("specfem::compute::element_types::boundary_tags", nspec) {

  for (int ispec = 0; ispec < nspec; ispec++) {
    const int ispec_mesh = mapping.compute_to_mesh(ispec);
    medium_tags(ispec) = tags.tags_container(ispec_mesh).medium_tag;
    property_tags(ispec) = tags.tags_container(ispec_mesh).property_tag;
    boundary_tags(ispec) = tags.tags_container(ispec_mesh).boundary_tag;
  }
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium_tag) const {

  int count = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (medium_tags(ispec) == medium_tag) {
      count++;
    }
  }

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements(
      "specfem::compute::element_types::get_elements_on_host", count);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (medium_tags(ispec) == medium_tag) {
      elements(index) = ispec;
      index++;
    }
  }

  return elements;
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium_tag) const {

  const auto dummy = get_elements_on_host(medium_tag);

  return Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(),
                                             dummy);
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

  int count = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (medium_tags(ispec) == medium_tag &&
        property_tags(ispec) == property_tag) {
      count++;
    }
  }

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements(
      "specfem::compute::element_types::get_elements_on_host", count);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (medium_tags(ispec) == medium_tag &&
        property_tags(ispec) == property_tag) {
      elements(index) = ispec;
      index++;
    }
  }

  return elements;
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

  const auto dummy = get_elements_on_host(medium_tag, property_tag);

  return Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(),
                                             dummy);
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag,
    const specfem::element::boundary_tag boundary_tag) const {

  int count = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (medium_tags(ispec) == medium_tag &&
        property_tags(ispec) == property_tag &&
        boundary_tags(ispec) == boundary_tag) {
      count++;
    }
  }

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements(
      "specfem::compute::element_types::get_elements_on_host", count);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (medium_tags(ispec) == medium_tag &&
        property_tags(ispec) == property_tag &&
        boundary_tags(ispec) == boundary_tag) {
      elements(index) = ispec;
      index++;
    }
  }

  return elements;
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag,
    const specfem::element::boundary_tag boundary_tag) const {

  const auto dummy =
      get_elements_on_host(medium_tag, property_tag, boundary_tag);

  return Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(),
                                             dummy);
}
