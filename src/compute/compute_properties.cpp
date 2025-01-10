#include "compute/properties/properties.hpp"

specfem::compute::properties::properties(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::mesh::materials &materials)
    : specfem::compute::impl::element_types(nspec, ngllz, ngllx, mapping,
                                            tags) {

  // compute total number of elastic and acoustic spectral elements
  int n_elastic_isotropic;
  int n_elastic_anisotropic;
  int n_acoustic;

  specfem::compute::impl::compute_number_of_elements_per_medium(
      nspec, mapping, tags, h_medium_tags, h_property_tags, n_elastic_isotropic,
      n_elastic_anisotropic, n_acoustic);

  acoustic_isotropic = specfem::medium::material_properties<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>(
      nspec, n_acoustic, ngllz, ngllx, mapping, tags, materials,
      h_property_index_mapping);

  elastic_isotropic = specfem::medium::material_properties<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>(
      nspec, n_elastic_isotropic, ngllz, ngllx, mapping, tags, materials,
      h_property_index_mapping);

  elastic_anisotropic = specfem::medium::material_properties<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic>(
      nspec, n_elastic_anisotropic, ngllz, ngllx, mapping, tags, materials,
      h_property_index_mapping);

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  Kokkos::deep_copy(medium_tags, h_medium_tags);
  Kokkos::deep_copy(property_tags, h_property_tags);

  return;
}
