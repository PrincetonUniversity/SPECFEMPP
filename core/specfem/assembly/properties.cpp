// Constructor definitions and explicit instantiation for properties<dim2/dim3>.
// domain_properties.tpp is included here so its constructor bodies are
// available for instantiation (kept out of the main header to limit
// compilation impact in other translation units).
#include "specfem/assembly/properties.hpp"
#include "specfem/assembly/impl/domain_properties.tpp"

template <specfem::element::dimension_tag DimensionTag>
specfem::assembly::properties<DimensionTag>::properties(
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::mesh::materials<DimensionTag> &materials,
    bool has_gll_model) {
  const auto &grid = element_types.element_grid;
  const int nspec = element_types.nspec;
  this->nspec = nspec;
  this->element_grid = grid;
  this->property_index_mapping =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
          "specfem::assembly::properties::property_index_mapping", nspec);
  this->h_property_index_mapping =
      Kokkos::create_mirror_view(this->property_index_mapping);
  for (int ispec = 0; ispec < nspec; ++ispec)
    this->h_property_index_mapping(ispec) = -1;
  specfem::tag_dispatch::for_each(this->combinations, [&]<typename TagsType>() {
    this->value
        .template get<TagsType>() = specfem::assembly::impl::domain_properties<
        TagsType::dimension_tag, TagsType::medium_tag, TagsType::property_tag>(
        element_types.get_elements_on_host(TagsType::medium_tag,
                                           TagsType::property_tag,
                                           TagsType::attenuation_tag),
        mesh, materials, has_gll_model, this->h_property_index_mapping);
  });
  Kokkos::deep_copy(this->property_index_mapping,
                    this->h_property_index_mapping);
}

template struct specfem::assembly::properties<
    specfem::element::dimension_tag::dim2>;
template struct specfem::assembly::properties<
    specfem::element::dimension_tag::dim3>;
