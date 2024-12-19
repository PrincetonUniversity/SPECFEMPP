#include "compute/impl/element_types.hpp"

specfem::compute::impl::element_types::element_types(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags)
    : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
      medium_tags("specfem::compute::properties::medium_tags", nspec),
      h_medium_tags(Kokkos::create_mirror_view(medium_tags)),
      property_index_mapping(
          "specfem::compute::properties::property_index_mapping", nspec),
      property_tags("specfem::compute::properties::property_tags", nspec),
      h_property_tags(Kokkos::create_mirror_view(property_tags)),
      h_property_index_mapping(
          Kokkos::create_mirror_view(property_index_mapping)) {
  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  Kokkos::deep_copy(medium_tags, h_medium_tags);
  Kokkos::deep_copy(property_tags, h_property_tags);
  return;
}

Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::compute::impl::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium) const {

  const int nspec = this->nspec;

  int nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (h_medium_tags(ispec) == medium) {
      nelements++;
    }
  }

  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> elements(
      "specfem::compute::properties::get_elements_on_host", nelements);

  nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (h_medium_tags(ispec) == medium) {
      elements(nelements) = ispec;
      nelements++;
    }
  }

  return elements;
}

Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
specfem::compute::impl::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium) const {

  // If the elements have not been computed, compute them.
  // The elements need to be computed in serial on the host.
  // This function computes the host elements on host and then
  // copies them to the device.
  const auto host_elements = this->get_elements_on_host(medium);

  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
      elements("specfem::compute::properties::get_elements_on_device",
               host_elements.extent(0));

  Kokkos::deep_copy(elements, host_elements);

  return elements;
}

Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::compute::impl::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property) const {

  const int nspec = this->nspec;

  int nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (h_medium_tags(ispec) == medium &&
        h_property_tags(ispec) == property) {
      nelements++;
    }
  }

  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> elements(
      "specfem::compute::properties::get_elements_on_host", nelements);

  nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (h_medium_tags(ispec) == medium &&
        h_property_tags(ispec) == property) {
      elements(nelements) = ispec;
      nelements++;
    }
  }

  return elements;
}

Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
specfem::compute::impl::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property) const {

  // If the elements have not been computed, compute them.
  // The elements need to be computed in serial on the host.
  // This function computes the host elements on host and then
  // copies them to the device.
  const auto host_elements = this->get_elements_on_host(medium, property);

  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
      elements("specfem::compute::properties::get_elements_on_device",
               host_elements.extent(0));

  Kokkos::deep_copy(elements, host_elements);

  return elements;
}
