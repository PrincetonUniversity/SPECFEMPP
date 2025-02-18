#include "compute/properties/properties.hpp"
#include "enumerations/dimension.hpp"

specfem::compute::properties::properties(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::element_types &element_types,
    const specfem::mesh::materials<specfem::dimension::type::dim2> &materials,
    const bool has_gll_model) {

  this->nspec = nspec;
  this->ngllz = ngllz;
  this->ngllx = ngllx;

  this->property_index_mapping =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
          "specfem::compute::properties::property_index_mapping", nspec);
  this->h_property_index_mapping =
      Kokkos::create_mirror_view(property_index_mapping);

  const auto elastic_isotropic_elements = element_types.get_elements_on_host(
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic);

  const auto elastic_anisotropic_elements = element_types.get_elements_on_host(
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic);

  const auto acoustic_elements = element_types.get_elements_on_host(
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic);

  for (int ispec = 0; ispec < nspec; ++ispec) {
    h_property_index_mapping(ispec) = -1;
  }

  acoustic_isotropic = specfem::medium::material_properties<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>(
      acoustic_elements, ngllz, ngllx, materials, has_gll_model,
      h_property_index_mapping);

  elastic_isotropic = specfem::medium::material_properties<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>(
      elastic_isotropic_elements, ngllz, ngllx, materials, has_gll_model,
      h_property_index_mapping);

  elastic_anisotropic = specfem::medium::material_properties<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic>(
      elastic_anisotropic_elements, ngllz, ngllx, materials, has_gll_model,
      h_property_index_mapping);

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);

  return;
}
