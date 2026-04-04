#include "specfem/assembly/element_types.hpp"

namespace {

// ---------------------------------------------------------------------------
// Generic helper: dispatch over a Storage, find the entry whose tags match
// the runtime predicate Pred, and return it.
// ---------------------------------------------------------------------------
template <typename ViewType, typename Storage, typename Combinations,
          typename Pred>
ViewType dispatch_get(const Storage &storage, Combinations combos, Pred &&pred,
                      const char *error_msg) {
  ViewType result;
  bool found = false;
  specfem::tag_dispatch::for_each(combos, [&]<typename TagsType>() {
    if (!found && pred.template operator()<TagsType>()) {
      result = storage.template get<TagsType>();
      found = true;
    }
  });
  if (!found)
    throw std::runtime_error(error_msg);
  return result;
}

} // namespace

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

  // Helper: allocate a device view + host mirror, fill from a predicate, sync.
  auto fill_views = [&](auto &dev_view, auto &host_view, const auto &label,
                        auto &&pred) {
    int count = 0;
    for (int ispec = 0; ispec < nspec; ispec++)
      if (pred(ispec))
        count++;

    dev_view = IndexViewType(label, count);
    host_view = Kokkos::create_mirror_view(dev_view);

    int index = 0;
    for (int ispec = 0; ispec < nspec; ispec++)
      if (pred(ispec))
        host_view(index++) = ispec;

    Kokkos::deep_copy(dev_view, host_view);
  };

  // 1. Index elements by medium.
  specfem::tag_dispatch::for_each(
      combinations_by_medium, [&]<typename TagsType>() {
        fill_views(elements_by_medium.template get<TagsType>(),
                   h_elements_by_medium.template get<TagsType>(),
                   "element_by_medium_" + TagsType::name(), [&](int ispec) {
                     return medium_tags(ispec) == TagsType::medium_tag;
                   });
      });

  // 2. Index elements by material (medium + property + attenuation).
  specfem::tag_dispatch::for_each(
      combinations_by_material, [&]<typename TagsType>() {
        fill_views(elements_by_material.template get<TagsType>(),
                   h_elements_by_material.template get<TagsType>(),
                   "element_by_material_" + TagsType::name(), [&](int ispec) {
                     return medium_tags(ispec) == TagsType::medium_tag &&
                            property_tags(ispec) == TagsType::property_tag &&
                            attenuation_tags(ispec) ==
                                TagsType::attenuation_tag;
                   });
      });

  // 3. Index elements by boundary (medium + property + boundary).
  specfem::tag_dispatch::for_each(
      combinations_by_boundary, [&]<typename TagsType>() {
        fill_views(elements_by_boundary.template get<TagsType>(),
                   h_elements_by_boundary.template get<TagsType>(),
                   "element_by_boundary_" + TagsType::name(), [&](int ispec) {
                     return medium_tags(ispec) == TagsType::medium_tag &&
                            property_tags(ispec) == TagsType::property_tag &&
                            boundary_tags(ispec) == TagsType::boundary_tag;
                   });
      });
}

// ---------------------------------------------------------------------------
// get_elements_on_host / get_elements_on_device — by medium
// ---------------------------------------------------------------------------
Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::assembly::element_types<specfem::element::dimension_tag::dim3>::
    get_elements_on_host(const specfem::element::medium_tag medium_tag) const {
  return dispatch_get<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >(
      h_elements_by_medium, combinations_by_medium,
      [medium_tag]<typename TagsType>() {
        return TagsType::medium_tag == medium_tag;
      },
      "Medium tag not found");
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::assembly::element_types<specfem::element::dimension_tag::dim3>::
    get_elements_on_device(
        const specfem::element::medium_tag medium_tag) const {
  return dispatch_get<Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(
      elements_by_medium, combinations_by_medium,
      [medium_tag]<typename TagsType>() {
        return TagsType::medium_tag == medium_tag;
      },
      "Medium tag not found");
}

// ---------------------------------------------------------------------------
// get_elements_on_host / get_elements_on_device — by material
// ---------------------------------------------------------------------------
Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> specfem::assembly::
    element_types<specfem::element::dimension_tag::dim3>::get_elements_on_host(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag,
        const specfem::element::attenuation_tag attenuation_tag) const {
  return dispatch_get<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >(
      h_elements_by_material, combinations_by_material,
      [medium_tag, property_tag, attenuation_tag]<typename TagsType>() {
        return TagsType::medium_tag == medium_tag &&
               TagsType::property_tag == property_tag &&
               TagsType::attenuation_tag == attenuation_tag;
      },
      "Medium tag or property tag not found");
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::assembly::element_types<specfem::element::dimension_tag::dim3>::
    get_elements_on_device(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag,
        const specfem::element::attenuation_tag attenuation_tag) const {
  return dispatch_get<Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(
      elements_by_material, combinations_by_material,
      [medium_tag, property_tag, attenuation_tag]<typename TagsType>() {
        return TagsType::medium_tag == medium_tag &&
               TagsType::property_tag == property_tag &&
               TagsType::attenuation_tag == attenuation_tag;
      },
      "Medium tag or property tag not found");
}

// ---------------------------------------------------------------------------
// get_elements_on_host / get_elements_on_device — by boundary
// ---------------------------------------------------------------------------
Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> specfem::assembly::
    element_types<specfem::element::dimension_tag::dim3>::get_elements_on_host(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag,
        const specfem::element::boundary_tag boundary_tag) const {
  return dispatch_get<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >(
      h_elements_by_boundary, combinations_by_boundary,
      [medium_tag, property_tag, boundary_tag]<typename TagsType>() {
        return TagsType::medium_tag == medium_tag &&
               TagsType::property_tag == property_tag &&
               TagsType::boundary_tag == boundary_tag;
      },
      "Medium tag, property tag or boundary tag not found");
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::assembly::element_types<specfem::element::dimension_tag::dim3>::
    get_elements_on_device(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag,
        const specfem::element::boundary_tag boundary_tag) const {
  return dispatch_get<Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(
      elements_by_boundary, combinations_by_boundary,
      [medium_tag, property_tag, boundary_tag]<typename TagsType>() {
        return TagsType::medium_tag == medium_tag &&
               TagsType::property_tag == property_tag &&
               TagsType::boundary_tag == boundary_tag;
      },
      "Medium tag, property tag or boundary tag not found");
}
