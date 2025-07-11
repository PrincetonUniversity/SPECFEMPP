#include "specfem/assembly/properties.hpp"
#include "enumerations/interface.hpp"
#include "medium/properties_container.hpp"
#include "medium/properties_container.tpp"
#include "specfem/assembly/mesh.hpp"

specfem::assembly::properties<specfem::dimension::type::dim2>::properties(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::mesh::materials<specfem::dimension::type::dim2> &materials,
    const bool has_gll_model) {

  this->nspec = nspec;
  this->ngllz = ngllz;
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
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      CAPTURE(value) {
        _value_ = specfem::medium::properties_container<
            _dimension_tag_, _medium_tag_, _property_tag_>(
            element_types.get_elements_on_host(_medium_tag_, _property_tag_),
            mesh, ngllz, ngllx, materials, has_gll_model,
            h_property_index_mapping);
      })

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);

  return;
}
