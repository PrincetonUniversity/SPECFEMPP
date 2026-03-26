#include "specfem/assembly/field_derivative_storage/dim3/field_derivative_storage.hpp"
#include <type_traits>

specfem::assembly::FieldDerivativeStorage<
    specfem::element::dimension_tag::dim3>::
    FieldDerivativeStorage(
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim3> &element_types,
        const int nspec_global, const int ngllz, const int nglly,
        const int ngllx) {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC),
       ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
      CAPTURE(fd_medium) {
        using medium_t = std::remove_reference_t<decltype(_fd_medium_)>;
        auto elements = element_types.get_elements_on_host(
            _medium_tag_, _property_tag_, _attenuation_tag_);
        _fd_medium_ = medium_t(elements, nspec_global, ngllz, nglly, ngllx);
      })
}
