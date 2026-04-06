#include "specfem/assembly/element_types.hpp"

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

  auto make_host_index_view = [&]<typename TagsType>() {
    constexpr bool has_medium = requires { TagsType::medium_tag; };
    constexpr bool has_property = requires { TagsType::property_tag; };
    constexpr bool has_attenuation = requires { TagsType::attenuation_tag; };
    constexpr bool has_boundary = requires { TagsType::boundary_tag; };

    std::string prefix;
    if constexpr (has_attenuation && !has_boundary) {
      prefix = "element_by_material_";
    } else if constexpr (!has_attenuation && has_boundary) {
      prefix = "element_by_boundary_";
    } else if constexpr (!has_attenuation && !has_boundary) {
      prefix = "element_by_medium_";
    } else {
      static_assert(!has_attenuation || !has_boundary,
                    "Unsupported tag combination for element index view");
    }

    auto matches_tags = [&](const int ispec) {
      bool match = medium_tags(ispec) == TagsType::medium_tag;
      if constexpr (has_attenuation) {
        match = match && (property_tags(ispec) == TagsType::property_tag) &&
                (attenuation_tags(ispec) == TagsType::attenuation_tag);
      }
      if constexpr (has_boundary) {
        match = match && (property_tags(ispec) == TagsType::property_tag) &&
                (boundary_tags(ispec) == TagsType::boundary_tag);
      }
      return match;
    };

    int count = 0;
    for (int ispec = 0; ispec < nspec; ispec++) {
      if (matches_tags(ispec))
        count++;
    }

    HostIndexViewType host_view(prefix + TagsType::name(), count);

    int index = 0;
    for (int ispec = 0; ispec < nspec; ispec++) {
      if (matches_tags(ispec))
        host_view(index++) = ispec;
    }

    return host_view;
  };

  // 1. Index elements by medium.
  h_elements_by_medium = { make_host_index_view };

  elements_by_medium = { [&]<typename TagsType>() {
    const auto host_view = h_elements_by_medium.template get<TagsType>();
    IndexViewType device_view("element_by_medium_" + TagsType::name(),
                              host_view.extent(0));
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
  } };

  // 2. Index elements by material (medium + property + attenuation).
  h_elements_by_material = { make_host_index_view };

  elements_by_material = { [&]<typename TagsType>() {
    const auto host_view = h_elements_by_material.template get<TagsType>();
    IndexViewType device_view("element_by_material_" + TagsType::name(),
                              host_view.extent(0));
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
  } };

  // 3. Index elements by boundary (medium + property + boundary).
  h_elements_by_boundary = { make_host_index_view };

  elements_by_boundary = { [&]<typename TagsType>() {
    const auto host_view = h_elements_by_boundary.template get<TagsType>();
    IndexViewType device_view("element_by_boundary_" + TagsType::name(),
                              host_view.extent(0));
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
  } };
}
