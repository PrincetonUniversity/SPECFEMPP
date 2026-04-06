#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <string>

namespace specfem::assembly::element_types_impl {

// ── Base: parameterised by the dimension and all five tag-set types.
//    Combinations, storage members, accessor methods, and the
//    index-store builder all live here so each dim-specific
//    specialisation only needs to pass the set types as template
//    arguments.

template <specfem::element::dimension_tag DimensionTag, auto MediumSet,
          auto PropertySet, auto BoundarySet, auto AttenuationSet>
struct element_types_base {
protected:
  template <typename T>
  using TagViewType = Kokkos::View<T *, Kokkos::DefaultHostExecutionSpace>;

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using HostIndexViewType =
      Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>;

  template <typename Sets>
  using IndexStorage = specfem::tag_dispatch::Storage<IndexViewType, Sets>;

  template <typename Sets>
  using HostIndexStorage =
      specfem::tag_dispatch::Storage<HostIndexViewType, Sets>;

public:
  static constexpr auto dimension_tag = DimensionTag;

  // ── Tag-combination sets ─────────────────────────────────────────────────

  static constexpr auto combinations_by_medium =
      specfem::tag_dispatch::dimension_set<DimensionTag>{} * MediumSet;
  static constexpr auto combinations_by_material =
      combinations_by_medium * PropertySet * AttenuationSet;
  static constexpr auto combinations_by_boundary =
      combinations_by_medium * PropertySet * BoundarySet;

  // ── Per-element tag views (host) ─────────────────────────────────────────

  int nspec{};
  specfem::mesh_entity::element_grid<DimensionTag> element_grid{};

  TagViewType<specfem::element::medium_tag> medium_tags;
  TagViewType<specfem::element::property_tag> property_tags;
  TagViewType<specfem::element::boundary_tag> boundary_tags;
  TagViewType<specfem::element::attenuation_tag> attenuation_tags;

protected:
  // ── Index stores ─────────────────────────────────────────────────────────

  IndexStorage<decltype(combinations_by_medium)> elements_by_medium;
  HostIndexStorage<decltype(combinations_by_medium)> h_elements_by_medium;
  IndexStorage<decltype(combinations_by_material)> elements_by_material;
  HostIndexStorage<decltype(combinations_by_material)> h_elements_by_material;
  IndexStorage<decltype(combinations_by_boundary)> elements_by_boundary;
  HostIndexStorage<decltype(combinations_by_boundary)> h_elements_by_boundary;

public:
  element_types_base() = default;

  element_types_base(
      int nspec,
      const specfem::mesh_entity::element_grid<DimensionTag> &element_grid,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::mesh::tags<DimensionTag> &tags)
      : nspec(nspec), element_grid(element_grid),
        medium_tags("specfem::assembly::element_types::medium_tags", nspec),
        property_tags("specfem::assembly::element_types::property_tags", nspec),
        boundary_tags("specfem::assembly::element_types::boundary_tags", nspec),
        attenuation_tags("specfem::assembly::element_types::attenuation_tags",
                         nspec) {
    for (int ispec = 0; ispec < nspec; ispec++) {
      int ispec_mesh;
      if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
        ispec_mesh = mesh.compute_to_mesh(ispec);
      } else {
        ispec_mesh = mesh.h_compute_to_mesh(ispec);
      }
      medium_tags(ispec) = tags.tags_container(ispec_mesh).medium_tag;
      property_tags(ispec) = tags.tags_container(ispec_mesh).property_tag;
      attenuation_tags(ispec) = tags.tags_container(ispec_mesh).attenuation_tag;
      boundary_tags(ispec) = tags.tags_container(ispec_mesh).boundary_tag;
    }

    auto make_host_index_view = [&]<typename TagsType>() -> HostIndexViewType {
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

      auto matches = [&](const int ispec) {
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
      for (int ispec = 0; ispec < nspec; ++ispec)
        if (matches(ispec))
          ++count;

      HostIndexViewType host_view(prefix + TagsType::name(), count);

      int index = 0;
      for (int ispec = 0; ispec < nspec; ++ispec)
        if (matches(ispec))
          host_view(index++) = ispec;

      return host_view;
    };

    auto make_device_storage = [&](auto &h_storage) {
      return [&h_storage]<typename TagsType>() -> IndexViewType {
        const auto host_view = h_storage.template get<TagsType>();
        IndexViewType device_view(host_view.label(), host_view.extent(0));
        Kokkos::deep_copy(device_view, host_view);
        return device_view;
      };
    };

    // 1. Index by medium.
    h_elements_by_medium = { make_host_index_view };
    elements_by_medium = { make_device_storage(h_elements_by_medium) };

    // 2. Index by material (medium + property + attenuation).
    h_elements_by_material = { make_host_index_view };
    elements_by_material = { make_device_storage(h_elements_by_material) };

    // 3. Index by boundary (medium + property + boundary).
    h_elements_by_boundary = { make_host_index_view };
    elements_by_boundary = { make_device_storage(h_elements_by_boundary) };
  }

  // ── Accessors by medium ──────────────────────────────────────────────────

  HostIndexViewType
  get_elements_on_host(const specfem::element::medium_tag medium_tag) const {
    return h_elements_by_medium.get(medium_tag);
  }

  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag) const {
    return get_elements_on_host(medium_tag).extent(0);
  }

  IndexViewType
  get_elements_on_device(const specfem::element::medium_tag medium_tag) const {
    return elements_by_medium.get(medium_tag);
  }

  // ── Accessors by material (medium + property + attenuation) ─────────────

  HostIndexViewType get_elements_on_host(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag) const {
    return h_elements_by_material.get(medium_tag, property_tag,
                                      attenuation_tag);
  }

  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag) const {
    return get_elements_on_host(medium_tag, property_tag, attenuation_tag)
        .extent(0);
  }

  IndexViewType get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag) const {
    return elements_by_material.get(medium_tag, property_tag, attenuation_tag);
  }

  // ── Accessors by boundary (medium + property + boundary) ────────────────

  HostIndexViewType get_elements_on_host(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return h_elements_by_boundary.get(medium_tag, property_tag, boundary_tag);
  }

  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return get_elements_on_host(medium_tag, property_tag, boundary_tag)
        .extent(0);
  }

  IndexViewType get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return elements_by_boundary.get(medium_tag, property_tag, boundary_tag);
  }

  // ── Per-element tag accessors ────────────────────────────────────────────

  specfem::element::medium_tag get_medium_tag(const int ispec) const {
    return medium_tags(ispec);
  }

  specfem::element::property_tag get_property_tag(const int ispec) const {
    return property_tags(ispec);
  }

  specfem::element::boundary_tag get_boundary_tag(const int ispec) const {
    return boundary_tags(ispec);
  }

  specfem::element::attenuation_tag get_attenuation_tag(const int ispec) const {
    return attenuation_tags(ispec);
  }
};

} // namespace specfem::assembly::element_types_impl
