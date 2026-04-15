#include "specfem/assembly/kernels.hpp"

template <specfem::element::dimension_tag DimensionTag>
specfem::assembly::kernels<DimensionTag>::kernels(
    const specfem::assembly::element_types<DimensionTag> &element_types) {
  const auto &grid = element_types.element_grid;
  const int nspec = element_types.nspec;

  this->nspec = nspec;
  this->element_grid = grid;

  this->property_index_mapping =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
          "specfem::assembly::kernels::property_index_mapping", nspec);
  this->h_property_index_mapping =
      Kokkos::create_mirror_view(this->property_index_mapping);

  for (int ispec = 0; ispec < nspec; ++ispec) {
    this->h_property_index_mapping(ispec) = -1;
  }

  specfem::tag_dispatch::for_each(this->combinations, [&]<typename TagsType>() {
    this->value
        .template get<TagsType>() = specfem::assembly::impl::domain_kernels<
        TagsType::dimension_tag, TagsType::medium_tag, TagsType::property_tag>(
        element_types.get_elements_on_host(TagsType::medium_tag,
                                           TagsType::property_tag,
                                           TagsType::attenuation_tag),
        grid, this->h_property_index_mapping);
  });

  Kokkos::deep_copy(this->property_index_mapping,
                    this->h_property_index_mapping);
}

template struct specfem::assembly::kernels<
    specfem::element::dimension_tag::dim2>;
template struct specfem::assembly::kernels<
    specfem::element::dimension_tag::dim3>;
