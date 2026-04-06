#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/element_types/impl.hpp"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
specfem::assembly::element_types<specfem::element::dimension_tag::dim3>::
    element_types(
        const int nspec, const int ngllz, const int nglly, const int ngllx,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh,
        const specfem::mesh::tags<specfem::element::dimension_tag::dim3> &tags)
    : nspec(nspec),
      medium_tags("specfem::assembly::element_types::medium_tags", nspec),
      property_tags("specfem::assembly::element_types::property_tags", nspec),
      attenuation_tags("specfem::assembly::element_types::attenuation_tags",
                       nspec),
      boundary_tags("specfem::assembly::element_types::boundary_tags", nspec) {

  for (int ispec = 0; ispec < nspec; ispec++) {
    const int ispec_mesh = mesh.h_compute_to_mesh(ispec);
    medium_tags(ispec) = tags.tags_container(ispec_mesh).medium_tag;
    property_tags(ispec) = tags.tags_container(ispec_mesh).property_tag;
    attenuation_tags(ispec) = tags.tags_container(ispec_mesh).attenuation_tag;
    boundary_tags(ispec) = tags.tags_container(ispec_mesh).boundary_tag;
  }

  // 1. Index elements by medium.
  specfem::tag_dispatch::for_each(
      combinations_by_medium, [&]<typename TagsType>() {
        specfem::assembly::element_types_impl::fill_index_views(
            elements_by_medium.template get<TagsType>(),
            h_elements_by_medium.template get<TagsType>(),
            "element_by_medium_" + TagsType::name(), nspec, [&](int ispec) {
              return medium_tags(ispec) == TagsType::medium_tag;
            });
      });

  // 2. Index elements by material (medium + property + attenuation).
  specfem::tag_dispatch::for_each(
      combinations_by_material, [&]<typename TagsType>() {
        specfem::assembly::element_types_impl::fill_index_views(
            elements_by_material.template get<TagsType>(),
            h_elements_by_material.template get<TagsType>(),
            "element_by_material_" + TagsType::name(), nspec, [&](int ispec) {
              return medium_tags(ispec) == TagsType::medium_tag &&
                     property_tags(ispec) == TagsType::property_tag &&
                     attenuation_tags(ispec) == TagsType::attenuation_tag;
            });
      });

  // 3. Index elements by boundary (medium + property + boundary).
  specfem::tag_dispatch::for_each(
      combinations_by_boundary, [&]<typename TagsType>() {
        specfem::assembly::element_types_impl::fill_index_views(
            elements_by_boundary.template get<TagsType>(),
            h_elements_by_boundary.template get<TagsType>(),
            "element_by_boundary_" + TagsType::name(), nspec, [&](int ispec) {
              return medium_tags(ispec) == TagsType::medium_tag &&
                     property_tags(ispec) == TagsType::property_tag &&
                     boundary_tags(ispec) == TagsType::boundary_tag;
            });
      });
}
