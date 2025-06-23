#include "compute/properties/properties.hpp"
#include "enumerations/dimension.hpp"

specfem::compute::properties::properties(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::element_types &element_types,
    const specfem::compute::mesh_to_compute_mapping &mapping,
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

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      CAPTURE(value) {
        _value_ =
            specfem::medium::properties_container<_medium_tag_, _property_tag_>(
                element_types.get_elements_on_host(_medium_tag_,
                                                   _property_tag_),
                mapping, ngllz, ngllx, materials, has_gll_model,
                h_property_index_mapping);
      })

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);

  return;
}
