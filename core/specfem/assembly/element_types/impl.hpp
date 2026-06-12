#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/element.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/program/abort.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <sstream>

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

  using IndexRangeView = specfem::datatype::ElementIndexRange;

  // ── Tag-combination sets ─────────────────────────────────────────────────

  /** All valid (dimension, medium) combinations for this specialization. */
  static constexpr auto combinations_by_medium =
      specfem::tag_dispatch::dimension_set<ElementSets::dimension_tag>{} *
      ElementSets::medium_set;
  /** All valid (dimension, medium, property) combinations. */
  static constexpr auto combinations_by_medium_property =
      combinations_by_medium * ElementSets::property_set;
  /** All valid (dimension, medium, property, attenuation) combinations. */
  static constexpr auto combinations_by_material = combinations_by_medium *
                                                   ElementSets::property_set *
                                                   ElementSets::attenuation_set;
  /** All valid (dimension, medium, property, boundary) combinations. */
  static constexpr auto combinations_by_boundary = combinations_by_medium *
                                                   ElementSets::property_set *
                                                   ElementSets::boundary_set;
  /** All valid (dimension, medium, property, attenuation, boundary) combos. */
  static constexpr auto combinations_by_element =
      combinations_by_medium * ElementSets::property_set *
      ElementSets::attenuation_set * ElementSets::boundary_set;
  /** All valid (dimension, medium, property, attenuation, boundary, mpi)
   * combos. */
  static constexpr auto combinations_by_mpi_element =
      combinations_by_element * ElementSets::mpi_set;

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
  /** Host view of per-element MPI partition tags (size: nspec). */
  TagViewType<specfem::element::mpi_tag> mpi_tags;

protected:
  // ── Index stores ─────────────────────────────────────────────────────────
  // Each store holds one ElementIndexRange per valid tag combination.
  // Because mesh_to_compute_mapping guarantees contiguity, no Kokkos::View
  // allocation or host→device copy is needed.

  /** Index range store keyed by (dimension, medium). */
  specfem::tag_dispatch::Storage<IndexRangeView,
                                 decltype(combinations_by_medium)>
      elements_by_medium;
  /** Index range store keyed by (dimension, medium, property). */
  specfem::tag_dispatch::Storage<IndexRangeView,
                                 decltype(combinations_by_medium_property)>
      elements_by_medium_property;
  /** Index range store keyed by (dimension, medium, property, attenuation). */
  specfem::tag_dispatch::Storage<IndexRangeView,
                                 decltype(combinations_by_material)>
      elements_by_material;
  /** Index range store keyed by (dimension, medium, property, boundary). */
  specfem::tag_dispatch::Storage<IndexRangeView,
                                 decltype(combinations_by_boundary)>
      elements_by_boundary;
  /** Index range store keyed by (dimension, medium, property, attenuation,
   * boundary). */
  specfem::tag_dispatch::Storage<IndexRangeView,
                                 decltype(combinations_by_element)>
      elements_by_element;
  /** Index range store keyed by (dimension, medium, property, attenuation,
   * boundary, mpi). */
  specfem::tag_dispatch::Storage<IndexRangeView,
                                 decltype(combinations_by_mpi_element)>
      elements_by_mpi_element;

public:
  /** @brief Default constructor; leaves all views and stores empty. */
  element_types_base() = default;

  /**
   * @brief Construct from raw per-element tag views (no mesh indirection).
   *
   * Copies tags directly from the supplied host views (identity mapping) and
   * builds all index stores.
   *
   * @param nspec               Number of spectral elements.
   * @param element_grid        GLL grid layout (ngllx, ngllz, etc.).
   * @param medium_tags_in      Host view of per-element medium tags.
   * @param property_tags_in    Host view of per-element property tags.
   * @param attenuation_tags_in Host view of per-element attenuation tags.
   * @param boundary_tags_in    Host view of per-element boundary tags.
   */
  element_types_base(
      int nspec,
      const specfem::mesh_entity::element_grid<dimension_tag> &element_grid,
      const TagViewType<specfem::element::medium_tag> &medium_tags_in,
      const TagViewType<specfem::element::property_tag> &property_tags_in,
      const TagViewType<specfem::element::attenuation_tag> &attenuation_tags_in,
      const TagViewType<specfem::element::boundary_tag> &boundary_tags_in)
      : nspec(nspec), element_grid(element_grid),
        medium_tags("specfem::assembly::element_types::medium_tags", nspec),
        property_tags("specfem::assembly::element_types::property_tags", nspec),
        boundary_tags("specfem::assembly::element_types::boundary_tags", nspec),
        attenuation_tags("specfem::assembly::element_types::attenuation_tags",
                         nspec),
        mpi_tags("specfem::assembly::element_types::mpi_tags", nspec) {
    // mpi_tags is value-initialized to mpi_tag::inner (no MPI classification
    // from raw tag views).
    for (int ispec = 0; ispec < nspec; ispec++) {
      medium_tags(ispec) = medium_tags_in(ispec);
      property_tags(ispec) = property_tags_in(ispec);
      attenuation_tags(ispec) = attenuation_tags_in(ispec);
      boundary_tags(ispec) = boundary_tags_in(ispec);
    }
    build_index_stores();
  }

  /**
   * @brief Convenience constructor for 2D: accepts GLL counts as integers.
   *
   * Builds an @c element_grid from @p ngllz and @p ngllx, then delegates to
   * the tag-view constructor above.
   */
  element_types_base(
      int nspec, int ngllz, int ngllx,
      const TagViewType<specfem::element::medium_tag> &medium_tags_in,
      const TagViewType<specfem::element::property_tag> &property_tags_in,
      const TagViewType<specfem::element::attenuation_tag> &attenuation_tags_in,
      const TagViewType<specfem::element::boundary_tag> &boundary_tags_in)
      : element_types_base(
            nspec,
            specfem::mesh_entity::element_grid<dimension_tag>{ ngllz, ngllx },
            medium_tags_in, property_tags_in, attenuation_tags_in,
            boundary_tags_in) {}

  /**
   * @brief Convenience constructor for 3D: accepts GLL counts as integers.
   *
   * Builds an @c element_grid from @p ngllz, @p nglly, and @p ngllx, then
   * delegates to the tag-view constructor above.
   */
  element_types_base(
      int nspec, int ngllz, int nglly, int ngllx,
      const TagViewType<specfem::element::medium_tag> &medium_tags_in,
      const TagViewType<specfem::element::property_tag> &property_tags_in,
      const TagViewType<specfem::element::attenuation_tag> &attenuation_tags_in,
      const TagViewType<specfem::element::boundary_tag> &boundary_tags_in)
      : element_types_base(nspec,
                           specfem::mesh_entity::element_grid<dimension_tag>{
                               ngllz, nglly, ngllx },
                           medium_tags_in, property_tags_in,
                           attenuation_tags_in, boundary_tags_in) {}

  /**
   * @brief Construct from mesh objects, mapping compute-to-mesh indices.
   *
   * Translates each compute-domain index to its mesh-domain counterpart via
   * @p mesh, reads medium/property/boundary/attenuation/mpi tags from @p tags,
   * then builds all index stores (including the MPI-classified store). The
   * MPI inner/outer classification is carried by the mesh @p tags.
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
                         nspec),
        mpi_tags("specfem::assembly::element_types::mpi_tags", nspec) {
    for (int ispec = 0; ispec < nspec; ispec++) {
      const int ispec_mesh = mesh.h_compute_to_mesh(ispec);
      medium_tags(ispec) = tags.tags_container(ispec_mesh).medium_tag;
      property_tags(ispec) = tags.tags_container(ispec_mesh).property_tag;
      attenuation_tags(ispec) = tags.tags_container(ispec_mesh).attenuation_tag;
      boundary_tags(ispec) = tags.tags_container(ispec_mesh).boundary_tag;
      mpi_tags(ispec) = tags.tags_container(ispec_mesh).mpi_tag;
    }
    build_index_stores();
  }

private:
  void build_index_stores() {
    auto make_initializer = [&](auto... tag_views) {
      return [&, tag_views...]<typename TagsType>() -> IndexRangeView {
        int start = -1, end = 0;
        bool exited = false;
        for (int ispec = 0; ispec < nspec; ++ispec) {
          if (TagsType{}.has(tag_views(ispec)...)) {
            if (exited) {
              std::ostringstream msg;
              msg << "element_types: element combination is not contiguous in "
                     "compute-domain index space.\n"
                  << "  First block started at ispec=" << start
                  << " and ended before ispec=" << (end) << ".\n"
                  << "  A second matching element was found at ispec=" << ispec
                  << ".\n"
                  << "  This indicates a bug in mesh_to_compute_mapping: "
                     "attenuation, medium, property, boundary, and mpi tags "
                     "must all be included as sort keys so every tag "
                     "combination maps to a single contiguous range.";
              specfem::Logger::error(msg.str());
              specfem::program::abort();
            }
            if (start == -1)
              start = ispec;
            end = ispec + 1;
          } else if (start != -1) {
            exited = true;
          }
        }
        return (start == -1) ? IndexRangeView{ 0, 0 }
                             : IndexRangeView{ start, end };
      };
    };
    elements_by_medium = { make_initializer(medium_tags) };
    elements_by_medium_property = { make_initializer(medium_tags,
                                                     property_tags) };
    elements_by_material = { make_initializer(medium_tags, property_tags,
                                              attenuation_tags) };
    elements_by_boundary = { make_initializer(medium_tags, property_tags,
                                              boundary_tags) };
    elements_by_element = { make_initializer(medium_tags, property_tags,
                                             attenuation_tags, boundary_tags) };
    elements_by_mpi_element = { make_initializer(medium_tags, property_tags,
                                                 attenuation_tags,
                                                 boundary_tags, mpi_tags) };
  }

public:
  // ── Accessors by medium ──────────────────────────────────────────────────

  /**
   * @brief Element index range matching the given medium.
   * @param medium_tag Medium to query.
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView
  get_elements_on_host(const specfem::element::medium_tag medium_tag) const {
    return elements_by_medium.get(medium_tag);
  }

  /**
   * @brief Number of elements with the given medium.
   * @param medium_tag Medium to query.
   */
  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag) const {
    return get_elements_on_host(medium_tag).size();
  }

  /**
   * @brief Element index range matching the given medium (device-callable).
   * @param medium_tag Medium to query.
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView
  get_elements_on_device(const specfem::element::medium_tag medium_tag) const {
    return elements_by_medium.get(medium_tag);
  }

  // ── Accessors by medium + property ──────────────────────────────────────

  /**
   * @brief Element index range matching the given medium and property.
   * @param medium_tag   Medium to query.
   * @param property_tag Material property to query.
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView get_elements_on_host(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag) const {
    return elements_by_medium_property.get(medium_tag, property_tag);
  }

  /**
   * @brief Number of elements with the given medium and property.
   * @param medium_tag   Medium to query.
   * @param property_tag Material property to query.
   */
  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag) const {
    return get_elements_on_host(medium_tag, property_tag).size();
  }

  /**
   * @brief Element index range matching the given medium and property
   * (device-callable).
   * @param medium_tag   Medium to query.
   * @param property_tag Material property to query.
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag) const {
    return elements_by_medium_property.get(medium_tag, property_tag);
  }

  // ── Accessors by material (medium + property + attenuation) ─────────────

  /**
   * @brief Element index range matching the given material.
   * @param medium_tag      Medium to query.
   * @param property_tag    Material property to query.
   * @param attenuation_tag Attenuation model to query.
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView get_elements_on_host(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag) const {
    return elements_by_material.get(medium_tag, property_tag, attenuation_tag);
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
        .size();
  }

  /**
   * @brief Element index range matching the given material (device-callable).
   * @param medium_tag      Medium to query.
   * @param property_tag    Material property to query.
   * @param attenuation_tag Attenuation model to query.
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag) const {
    return elements_by_material.get(medium_tag, property_tag, attenuation_tag);
  }

  // ── Accessors by boundary (medium + property + boundary) ────────────────

  /**
   * @brief Element index range matching the given boundary condition.
   * @param medium_tag   Medium to query.
   * @param property_tag Material property to query.
   * @param boundary_tag Boundary condition to query.
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView get_elements_on_host(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return elements_by_boundary.get(medium_tag, property_tag, boundary_tag);
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
    return get_elements_on_host(medium_tag, property_tag, boundary_tag).size();
  }

  /**
   * @brief Element index range matching the given boundary condition
   * (device-callable).
   * @param medium_tag   Medium to query.
   * @param property_tag Material property to query.
   * @param boundary_tag Boundary condition to query.
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return elements_by_boundary.get(medium_tag, property_tag, boundary_tag);
  }

  // ── Accessors by element (medium + property + attenuation + boundary) ──────

  /**
   * @brief Element index range matching the given full element tag combination.
   */
  IndexRangeView get_elements_on_host(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return elements_by_element.get(medium_tag, property_tag, attenuation_tag,
                                   boundary_tag);
  }

  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return get_elements_on_host(medium_tag, property_tag, attenuation_tag,
                                boundary_tag)
        .size();
  }

  /**
   * @brief Element index range matching the given full element tag combination
   * (device-callable).
   */
  IndexRangeView get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag,
      const specfem::element::boundary_tag boundary_tag) const {
    return elements_by_element.get(medium_tag, property_tag, attenuation_tag,
                                   boundary_tag);
  }

  // ── Accessors by mpi element (medium + property + attenuation + boundary +
  // mpi) ──

  /**
   * @brief Element index range matching the given element type and MPI
   * partition classification.
   * @param medium_tag      Medium to query.
   * @param property_tag    Material property to query.
   * @param attenuation_tag Attenuation model to query.
   * @param boundary_tag    Boundary condition to query.
   * @param mpi_tag         MPI partition classification (inner or outer).
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView
  get_elements_on_host(const specfem::element::medium_tag medium_tag,
                       const specfem::element::property_tag property_tag,
                       const specfem::element::attenuation_tag attenuation_tag,
                       const specfem::element::boundary_tag boundary_tag,
                       const specfem::element::mpi_tag mpi_tag) const {
    return elements_by_mpi_element.get(medium_tag, property_tag,
                                       attenuation_tag, boundary_tag, mpi_tag);
  }

  /**
   * @brief Number of elements matching the given element type and MPI
   * partition classification.
   * @param medium_tag      Medium to query.
   * @param property_tag    Material property to query.
   * @param attenuation_tag Attenuation model to query.
   * @param boundary_tag    Boundary condition to query.
   * @param mpi_tag         MPI partition classification (inner or outer).
   */
  int get_number_of_elements(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag,
      const specfem::element::boundary_tag boundary_tag,
      const specfem::element::mpi_tag mpi_tag) const {
    return get_elements_on_host(medium_tag, property_tag, attenuation_tag,
                                boundary_tag, mpi_tag)
        .size();
  }

  /**
   * @brief Element index range matching the given element type and MPI
   * partition classification (device-callable).
   * @param medium_tag      Medium to query.
   * @param property_tag    Material property to query.
   * @param attenuation_tag Attenuation model to query.
   * @param boundary_tag    Boundary condition to query.
   * @param mpi_tag         MPI partition classification (inner or outer).
   * @return Contiguous half-open range [start, end) of spectral-element
   * indices.
   */
  IndexRangeView get_elements_on_device(
      const specfem::element::medium_tag medium_tag,
      const specfem::element::property_tag property_tag,
      const specfem::element::attenuation_tag attenuation_tag,
      const specfem::element::boundary_tag boundary_tag,
      const specfem::element::mpi_tag mpi_tag) const {
    return elements_by_mpi_element.get(medium_tag, property_tag,
                                       attenuation_tag, boundary_tag, mpi_tag);
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

  /**
   * @brief MPI partition tag of element @p ispec in the compute domain.
   * @param ispec Compute-domain element index.
   * @return inner if the element is not on an MPI boundary, outer otherwise.
   *         Returns inner if MPI tags have not been initialized.
   */
  specfem::element::mpi_tag get_mpi_tag(const int ispec) const {
    if (mpi_tags.extent(0) == 0)
      return specfem::element::mpi_tag::inner;
    return mpi_tags(ispec);
  }
};

} // namespace specfem::assembly::element_types_impl
