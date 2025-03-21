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

  for (int ispec = 0; ispec < nspec; ++ispec) {
    h_property_index_mapping(ispec) = -1;
  }

  CALL_CODE_FOR_ALL_MATERIAL_SYSTEMS(
      CAPTURE(value) WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
      _value_ =
          specfem::medium::properties_container<_medium_tag_, _property_tag_>(
              element_types.get_elements_on_host(_medium_tag_, _property_tag_),
              ngllz, ngllx, materials, has_gll_model,
              h_property_index_mapping););

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);

  return;
}
