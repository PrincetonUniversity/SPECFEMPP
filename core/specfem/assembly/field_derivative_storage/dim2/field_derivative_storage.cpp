#include "specfem/assembly/field_derivative_storage/dim2/field_derivative_storage.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/macros.hpp"
#include <type_traits>

specfem::assembly::FieldDerivativeStorage<
    specfem::element::dimension_tag::dim2>::
    FieldDerivativeStorage(
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim2> &element_types,
        const int nspec_global, const int ngllz, const int ngllx) {

  // Initialize storage for all CONSTANT_ISOTROPIC combinations.
  // NONE combinations are default-constructed (empty, no-op).
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV)), CAPTURE(fd_medium) {
        // Select elements in this medium that require field derivative storage
        // (e.g. are attenuating).
        std::vector<int> element_indices;

        auto elements = element_types.get_elements_on_host(_medium_tag_);

        for (int element_index = 0; element_index < elements.extent(0);
             ++element_index) {

          const int ispec = elements(element_index);

          // This expresion could be
          bool requires_storage = (element_types.attenuation_tags(ispec) !=
                                   specfem::element::attenuation_tag::none);

          // Store the global
          if (requires_storage) {
            element_indices.push_back(ispec);
          }
        }

        const size_t count = element_indices.size();
        Kokkos::View<int *, Kokkos::HostSpace> subset_elements(
            "subset_elements", count);
        for (size_t i = 0; i < count; ++i)
          subset_elements(i) = element_indices[i];

        _fd_medium_ = { subset_elements, nspec_global, ngllz, ngllx };
      })
}
