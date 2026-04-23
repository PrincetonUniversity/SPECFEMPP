#include "specfem/assembly/field_derivative_storage/dim2/field_derivative_storage.hpp"
#include "specfem/assembly/element_types.hpp"
#include <type_traits>

specfem::assembly::FieldDerivativeStorage<
    specfem::element::dimension_tag::dim2>::
    FieldDerivativeStorage(
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim2> &element_types,
        const int nspec_global, const int ngllz, const int ngllx) {

  specfem::tag_dispatch::for_each(
      field_derivative_medium_combinations, [&]<typename TagsType>() {
        std::vector<int> element_indices;

        auto elements =
            element_types.get_elements_on_host(TagsType::medium_tag);

        for (int element_index = 0; element_index < elements.extent(0);
             ++element_index) {

          const int ispec = elements(element_index);

          bool requires_storage = (element_types.attenuation_tags(ispec) !=
                                   specfem::element::attenuation_tag::none);

          if (requires_storage) {
            element_indices.push_back(ispec);
          }
        }

        const size_t count = element_indices.size();
        Kokkos::View<int *, Kokkos::HostSpace> subset_elements(
            "subset_elements", count);
        for (size_t i = 0; i < count; ++i)
          subset_elements(i) = element_indices[i];

        field_derivative_storage.template get<TagsType>() = { subset_elements,
                                                              nspec_global,
                                                              ngllz, ngllx };
      });
}
