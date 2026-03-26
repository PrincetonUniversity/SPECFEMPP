#include "specfem/assembly/field_derivative_storage/dim2/field_derivative_storage.hpp"
#include <type_traits>

specfem::assembly::FieldDerivativeStorage<
    specfem::element::dimension_tag::dim2>::
    FieldDerivativeStorage(
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim2> &element_types,
        const int nspec_global, const int ngllz, const int ngllx) {

  // Initialize storage for all CONSTANT_ISOTROPIC combinations.
  // NONE combinations are default-constructed (empty, no-op).
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
                       ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
                      CAPTURE(fd_medium) {
                        using medium_t =
                            std::remove_reference_t<decltype(_fd_medium_)>;
                        auto elements = element_types.get_elements_on_host(
                            _medium_tag_, _property_tag_, _attenuation_tag_);
                        _fd_medium_ =
                            medium_t(elements, nspec_global, ngllz, 0, ngllx);
                      })
}
