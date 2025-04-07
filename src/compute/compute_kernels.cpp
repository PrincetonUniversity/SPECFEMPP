#include "compute/kernels/kernels.hpp"

specfem::compute::kernels::kernels(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::element_types &element_types) {

  this->nspec = nspec;
  this->ngllz = ngllz;
  this->ngllx = ngllx;

  this->property_index_mapping =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
          "specfem::compute::kernels::property_index_mapping", nspec);

  this->h_property_index_mapping =
      Kokkos::create_mirror_view(property_index_mapping);

  for (int ispec = 0; ispec < nspec; ++ispec) {
    h_property_index_mapping(ispec) = -1;
  }

  FOR_EACH_MATERIAL_SYSTEM(
      FROM((DIMENSION_TAG_DIM2),
           (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC,
            MEDIUM_TAG_POROELASTIC),
           (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
      CAPTURE(value) {
        _value_ =
            specfem::medium::kernels_container<_medium_tag_, _property_tag_>(
                element_types.get_elements_on_host(_medium_tag_,
                                                   _property_tag_),
                ngllz, ngllx, h_property_index_mapping);
      })

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);

  return;
}
