#include "compute/kernels/kernels.hpp"

// Explicit template instantiation

template class specfem::compute::impl::kernels::material_kernels<
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic>;

template class specfem::compute::impl::kernels::material_kernels<
    specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic>;

namespace {
void compute_number_of_elements_per_medium(
    const int nspec, const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags &tags,
    const specfem::kokkos::HostView1d<specfem::element::medium_tag>
        &h_element_types,
    const specfem::kokkos::HostView1d<specfem::element::property_tag>
        &h_element_property,
    int &n_elastic, int &n_acoustic) {

  Kokkos::parallel_reduce(
      "specfem::compute::properties::compute_number_of_elements_per_medium",
      specfem::kokkos::HostRange(0, nspec),
      [=](const int ispec, int &n_elastic, int &n_acoustic) {
        const int ispec_mesh = mapping.compute_to_mesh(ispec);
        if (tags.tags_container(ispec_mesh).medium_tag ==
            specfem::element::medium_tag::elastic) {
          n_elastic++;
          h_element_types(ispec) = specfem::element::medium_tag::elastic;
          if (tags.tags_container(ispec_mesh).property_tag ==
              specfem::element::property_tag::isotropic) {
            h_element_property(ispec) =
                specfem::element::property_tag::isotropic;
          } else {
            throw std::runtime_error("Unknown property tag");
          }
        } else if (tags.tags_container(ispec_mesh).medium_tag ==
                   specfem::element::medium_tag::acoustic) {
          n_acoustic++;
          h_element_types(ispec) = specfem::element::medium_tag::acoustic;
          if (tags.tags_container(ispec_mesh).property_tag ==
              specfem::element::property_tag::isotropic) {
            h_element_property(ispec) =
                specfem::element::property_tag::isotropic;
          } else {
            throw std::runtime_error("Unknown property tag");
          }
        }
      },
      n_elastic, n_acoustic);

  if (n_elastic + n_acoustic != nspec)
    throw std::runtime_error("Number of elements per medium does not match "
                             "total number of elements");

  return;
}
} // namespace

specfem::compute::kernels::kernels(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags &tags)
    : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
      element_types("specfem::compute::properties::element_types", nspec),
      h_element_types(Kokkos::create_mirror_view(element_types)),
      property_index_mapping(
          "specfem::compute::properties::property_index_mapping", nspec),
      element_property("specfem::compute::properties::element_property", nspec),
      h_element_property(Kokkos::create_mirror_view(element_property)),
      h_property_index_mapping(
          Kokkos::create_mirror_view(property_index_mapping)) {
  // compute total number of elastic and acoustic spectral elements
  int n_elastic;
  int n_acoustic;

  compute_number_of_elements_per_medium(nspec, mapping, tags, h_element_types,
                                        h_element_property, n_elastic,
                                        n_acoustic);

  acoustic_isotropic = specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>(
      nspec, n_acoustic, ngllz, ngllx, mapping, tags, h_property_index_mapping);

  elastic_isotropic = specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>(
      nspec, n_elastic, ngllz, ngllx, mapping, tags, h_property_index_mapping);

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  Kokkos::deep_copy(element_types, h_element_types);
  Kokkos::deep_copy(element_property, h_element_property);

  return;
}
