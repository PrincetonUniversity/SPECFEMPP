#include "specfem/assembly/kernels.hpp"

specfem::assembly::kernels<specfem::dimension::type::dim2>::kernels(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types) {

  this->nspec = nspec;
  this->ngllz = ngllz;
  this->ngllx = ngllx;

  this->property_index_mapping =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
          "specfem::assembly::kernels::property_index_mapping", nspec);

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
        _value_ = specfem::medium::kernels_container<
            _dimension_tag_, _medium_tag_, _property_tag_>(
            element_types.get_elements_on_host(_medium_tag_, _property_tag_),
            ngllz, ngllx, h_property_index_mapping);
      })

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);

  return;
}
