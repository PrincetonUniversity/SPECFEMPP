#include "specfem/assembly/properties.hpp"
#include "specfem/assembly/impl/domain_properties.hpp"
#include "specfem/assembly/impl/domain_properties.tpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"

specfem::assembly::properties<specfem::element::dimension_tag::dim3>::
    properties(
        const int nspec, const int ngllz, const int nglly, const int ngllx,
        const specfem::mesh::materials<dimension_tag> &materials,
        const specfem::assembly::element_types<dimension_tag> &element_types) {
  this->nspec = nspec;
  this->ngllz = ngllz;
  this->nglly = nglly;
  this->ngllx = ngllx;

  this->property_index_mapping =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
          "specfem::assembly::properties::property_index_mapping", nspec);
  this->h_property_index_mapping =
      Kokkos::create_mirror_view(property_index_mapping);

  for (int ispec = 0; ispec < nspec; ++ispec) {
    h_property_index_mapping(ispec) = -1;
  }

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC),
       PROPERTY_TAG(ISOTROPIC), ATTENUATION_TAG(NONE, ISOTROPIC_CONSTANT)),
      CAPTURE(value) {
        _value_ = specfem::assembly::impl::domain_properties<
            _dimension_tag_, _medium_tag_, _property_tag_>(
            element_types.get_elements_on_host(_medium_tag_, _property_tag_,
                                               _attenuation_tag_),
            nspec, ngllz, nglly, ngllx, materials, h_property_index_mapping);
      })

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  return;
}
