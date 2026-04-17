#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <string>

namespace specfem::assembly::element_types_impl {

/**
 * @brief Base class for element-type classification and index storage.
 *
 * Parameterised by the dimension tag and the sets of valid medium, property,
 * boundary, and attenuation tags. Holds compile-time tag-combination sets,
 * per-element tag views, index stores, and all accessor methods. Each
 * dimension-specific specialisation supplies the tag-set values as template
 * arguments and inherits the full API via ``using base_type::base_type``.
 *
 * @tparam DimensionTag  Dimension of the simulation (dim2 or dim3).
 * @tparam MediumSet     Set of valid medium tags for this dimension.
 * @tparam PropertySet   Set of valid property tags.
 * @tparam BoundarySet   Set of valid boundary tags.
 * @tparam AttenuationSet Set of valid attenuation tags.
 */
template <typename ElementSets> struct element_types_base {
public:
  /** Dimension tag for this specialization (dim2 or dim3). */
  static constexpr auto dimension_tag = ElementSets::dimension_tag;

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

  // ── Tag-combination sets ─────────────────────────────────────────────────

  /** All valid (dimension, medium) combinations for this specialization. */
  static constexpr auto combinations_by_medium =
      specfem::tag_dispatch::dimension_set<ElementSets::dimension_tag>{} *
      ElementSets::medium_set;
  /** All valid (dimension, medium, property, attenuation) combinations. */
  static constexpr auto combinations_by_material = combinations_by_medium *
                                                   ElementSets::property_set *
                                                   ElementSets::attenuation_set;
  /** All valid (dimension, medium, property, boundary) combinations. */
  static constexpr auto combinations_by_boundary = combinations_by_medium *
                                                   ElementSets::property_set *
                                                   ElementSets::boundary_set;

public:
  // ── Per-element tag views (host) ─────────────────────────────────────────

  /** Number of spectral elements in the compute domain. */
  int nspec{};
  /** GLL grid layout for this dimension (ngllx, ngllz, etc.). */
  specfem::mesh_entity::element_grid<ElementSets::dimension_tag> element_grid{};

  /** Host view of per-element medium tags (size: nspec). */
  TagViewType<specfem::element::medium_tag> medium_tags;
  /** Host view of per-element property tags (size: nspec). */
  TagViewType<specfem::element::property_tag> property_tags;
  /** Host view of per-element boundary tags (size: nspec). */
  TagViewType<specfem::element::boundary_tag> boundary_tags;
  /** Host view of per-element attenuation tags (size: nspec). */
  TagViewType<specfem::element::attenuation_tag> attenuation_tags;

protected:
  // ── Index stores ─────────────────────────────────────────────────────────

  /** Device index store keyed by (dimension, medium). */
  IndexStorage<decltype(combinations_by_medium)> elements_by_medium;
  /** Host mirror of elements_by_medium. */
  HostIndexStorage<decltype(combinations_by_medium)> h_elements_by_medium;
  /** Device index store keyed by (dimension, medium, property, attenuation). */
  IndexStorage<decltype(combinations_by_material)> elements_by_material;
  /** Host mirror of elements_by_material. */
  HostIndexStorage<decltype(combinations_by_material)> h_elements_by_material;
  /** Device index store keyed by (dimension, medium, property, boundary). */
  IndexStorage<decltype(combinations_by_boundary)> elements_by_boundary;
  /** Host mirror of elements_by_boundary. */
  HostIndexStorage<decltype(combinations_by_boundary)> h_elements_by_boundary;

public:
  /** @brief Default constructor; leaves all views and stores empty. */
  element_types_base() = default;

  /**
   * @brief Construct and populate all per-element tag views and index stores.
   *
   * Maps compute-domain element indices to mesh-domain indices using
   * @p mesh, reads medium/property/boundary/attenuation tags from @p tags,
   * then builds and deep-copies all six index stores (host + device for each
   * of medium, material, and boundary granularities).
   *
   * @param nspec        Number of spectral elements in the compute domain.
   * @param element_grid GLL grid layout (ngllx, ngllz, etc.).
   * @param mesh         Compute-to-mesh index mapping.
   * @param tags         Per-element tag data from the mesh.
   */
  element_types_base(
      int nspec,
      const specfem::mesh_entity::element_grid<dimension_tag> &element_grid,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::mesh::tags<dimension_tag> &tags)
      : nspec(nspec), element_grid(element_grid),
        medium_tags("specfem::assembly::element_types::medium_tags", nspec),
        property_tags("specfem::assembly::element_types::property_tags", nspec),
        boundary_tags("specfem::assembly::element_types::boundary_tags", nspec),
        attenuation_tags("specfem::assembly::element_types::attenuation_tags",
                         nspec) {

    // ── Step 1: populate per-element tag views ──────────────────────────────
    // Translate each compute-domain index to its mesh-domain counterpart, then
    // copy the four tag values from the mesh tag container into the
    // corresponding host views.
    for (int ispec = 0; ispec < nspec; ispec++) {
      const int ispec_mesh = mesh.h_compute_to_mesh(ispec);
      medium_tags(ispec) = tags.tags_container(ispec_mesh).medium_tag;
      property_tags(ispec) = tags.tags_container(ispec_mesh).property_tag;
      attenuation_tags(ispec) = tags.tags_container(ispec_mesh).attenuation_tag;
      boundary_tags(ispec) = tags.tags_container(ispec_mesh).boundary_tag;
    }

    // ── Step 2: define index-building helpers ───────────────────────────────
    // make_index_views(matches) returns a functor that, for each TagsType,
    // loops over all nspec elements and gathers the indices of those matching
    // the criteria defined by the callable matches. The matches callable is
    // templated on TagsType and takes an int ispec argument.
    auto make_index_views = [&](auto matches) {
      return [&, matches]<typename TagsType>() {
        int count = 0;
        for (int ispec = 0; ispec < nspec; ++ispec)
          if (matches.template operator()<TagsType>(ispec))
            ++count;
        HostIndexViewType host_view("element_by_combination", count);
        int index = 0;
        for (int ispec = 0; ispec < nspec; ++ispec)
          if (matches.template operator()<TagsType>(ispec))
            host_view(index++) = ispec;
        return host_view;
      };
    };

    // 2.1.1 Index by (dimension, medium) only.
    h_elements_by_medium = { make_index_views(
        [&]<typename TagsType>(int ispec) {
          return TagsType::medium_tag == medium_tags(ispec);
        }) };

    // 2.1.2 Mirror host storage to device.
    elements_by_medium = specfem::tag_dispatch::create_mirror_storage_and_copy(
        Kokkos::DefaultExecutionSpace{}, h_elements_by_medium);

    // 2.2.1 Index by (dimension, medium, property, attenuation).
    h_elements_by_material = { make_index_views(
        [&]<typename TagsType>(int ispec) {
          return TagsType::medium_tag == medium_tags(ispec) &&
                 TagsType::property_tag == property_tags(ispec) &&
                 TagsType::attenuation_tag == attenuation_tags(ispec);
        }) };

    // 2.2.2 Mirror host storage to device.
    elements_by_material =
        specfem::tag_dispatch::create_mirror_storage_and_copy(
            Kokkos::DefaultExecutionSpace{}, h_elements_by_material);

    // 2.3.1 Index by (dimension, medium, property, boundary).
    h_elements_by_boundary = { make_index_views(
        [&]<typename TagsType>(int ispec) {
          return TagsType::medium_tag == medium_tags(ispec) &&
                 TagsType::property_tag == property_tags(ispec) &&
                 TagsType::boundary_tag == boundary_tags(ispec);
        }) };

    // 2.3.2 Mirror host storage to device.
    elements_by_boundary =
        specfem::tag_dispatch::create_mirror_storage_and_copy(
            Kokkos::DefaultExecutionSpace{}, h_elements_by_boundary);
  }

  // ── Accessors by medium ──────────────────────────────────────────────────

  /**
   * @brief Host view of element indices matching the given medium.
   * @param medium_tag Medium to query.
   * @return Host-accessible 1-D view of spectral-element indices.
   */
  HostIndexViewType
  get_elements_on_host(const specfem::element::medium_tag medium_tag) const {
    return h_elements_by_medium.get(medium_tag);
  }

  /**
   * @brief Number of elements with the given medium.
   * @param medium_tag Medium to query.
   */
  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag) const {
    return get_elements_on_host(medium_tag).extent(0);
  }

  /**
   * @brief Device view of element indices matching the given medium.
   * @param medium_tag Medium to query.
   * @return Device-accessible 1-D view of spectral-element indices.
   */
  IndexViewType
  get_elements_on_device(const specfem::element::medium_tag medium_tag) const {
    return elements_by_medium.get(medium_tag);
  }

  // ── Accessors by material (medium + property + attenuation) ─────────────

  /**
   * @brief Host view of element indices matching the given material.
   * @param medium_tag      Medium to query.
   * @param property_tag    Material property to query.
   * @param attenuation_tag Attenuation model to query.
   * @return Host-accessible 1-D view of spectral-element indices.
   */
  HostIndexViewType get_elements_on_host(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag) const {
    return h_elements_by_material.get(medium_tag, property_tag,
                                      attenuation_tag);
  }

  /**
   * @brief Number of elements with the given material.
   * @param medium_tag      Medium to query.
   * @param property_tag    Material property to query.
   * @param attenuation_tag Attenuation model to query.
   */
  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag) const {
    return get_elements_on_host(medium_tag, property_tag, attenuation_tag)
        .extent(0);
  }

  /**
   * @brief Device view of element indices matching the given material.
   * @param medium_tag      Medium to query.
   * @param property_tag    Material property to query.
   * @param attenuation_tag Attenuation model to query.
   * @return Device-accessible 1-D view of spectral-element indices.
   */
  IndexViewType get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag) const {
    return elements_by_material.get(medium_tag, property_tag, attenuation_tag);
  }

  // ── Accessors by boundary (medium + property + boundary) ────────────────

  /**
   * @brief Host view of element indices matching the given boundary condition.
   * @param medium_tag   Medium to query.
   * @param property_tag Material property to query.
   * @param boundary_tag Boundary condition to query.
   * @return Host-accessible 1-D view of spectral-element indices.
   */
  HostIndexViewType get_elements_on_host(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return h_elements_by_boundary.get(medium_tag, property_tag, boundary_tag);
  }

  /**
   * @brief Number of elements with the given boundary condition.
   * @param medium_tag   Medium to query.
   * @param property_tag Material property to query.
   * @param boundary_tag Boundary condition to query.
   */
  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return get_elements_on_host(medium_tag, property_tag, boundary_tag)
        .extent(0);
  }

  /**
   * @brief Device view of element indices matching the given boundary
   * condition.
   * @param medium_tag   Medium to query.
   * @param property_tag Material property to query.
   * @param boundary_tag Boundary condition to query.
   * @return Device-accessible 1-D view of spectral-element indices.
   */
  IndexViewType get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return elements_by_boundary.get(medium_tag, property_tag, boundary_tag);
  }

  // ── Per-element tag accessors ────────────────────────────────────────────

  /**
   * @brief Medium tag of element @p ispec in the compute domain.
   * @param ispec Compute-domain element index.
   */
  specfem::element::medium_tag get_medium_tag(const int ispec) const {
    return medium_tags(ispec);
  }

  /**
   * @brief Property tag of element @p ispec in the compute domain.
   * @param ispec Compute-domain element index.
   */
  specfem::element::property_tag get_property_tag(const int ispec) const {
    return property_tags(ispec);
  }

  /**
   * @brief Boundary tag of element @p ispec in the compute domain.
   * @param ispec Compute-domain element index.
   */
  specfem::element::boundary_tag get_boundary_tag(const int ispec) const {
    return boundary_tags(ispec);
  }

  /**
   * @brief Attenuation tag of element @p ispec in the compute domain.
   * @param ispec Compute-domain element index.
   */
  specfem::element::attenuation_tag get_attenuation_tag(const int ispec) const {
    return attenuation_tags(ispec);
  }
};

} // namespace specfem::assembly::element_types_impl
