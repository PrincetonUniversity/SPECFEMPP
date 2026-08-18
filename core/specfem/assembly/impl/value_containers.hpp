#pragma once

#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::impl {

/**
 * @brief Tag-set specifications for value_containers per dimension.
 *
 * Mirrors the ElementSets pattern from element_types.hpp, enumerating
 * only the valid (medium, property) combinations for which
 * value containers (kernels, properties) are instantiated.
 */
template <specfem::element::dimension_tag DimensionTag> struct ContainerSets;

template <> struct ContainerSets<specfem::element::dimension_tag::dim2> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;
  constexpr static auto combinations =
      DIMENSION_SET(dim2) *
      MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                 elastic_psv_t) *
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat);
};

template <> struct ContainerSets<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr static auto combinations = DIMENSION_SET(dim3) *
                                       MEDIUM_SET(elastic, acoustic) *
                                       PROPERTY_SET(isotropic);
};

/**
 * @brief Values for every quadrature point in the finite element mesh
 *
 * @tparam DimensionTag       Dimension tag selecting the ContainerSets
 *                            specialisation.
 * @tparam containers_type    3-parameter class template
 *                            (dimension_tag, medium_tag, property_tag).
 */
template <
    specfem::element::dimension_tag DimensionTag,
    template <specfem::element::dimension_tag, specfem::element::medium_tag,
              specfem::element::property_tag> class containers_type>
struct value_containers_base {

  using ContainerSetsType = ContainerSets<DimensionTag>;

  int nspec; ///< Total number of spectral elements
  specfem::mesh_entity::element_grid<DimensionTag> element_grid; ///< GLL grid
                                                                 ///< layout

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto combinations = ContainerSetsType::combinations;

  /// Per-(medium,property) element index range; enables arithmetic index
  /// mapping without a device View.
  specfem::tag_dispatch::Storage<specfem::datatype::ElementIndexRange,
                                 decltype(combinations)>
      element_ranges;

  template <typename TagsType>
  using ContainerType =
      containers_type<TagsType::dimension_tag, TagsType::medium_tag,
                      TagsType::property_tag>;

  specfem::tag_dispatch::TypedStorage<ContainerType, decltype(combinations)>
      value;

  /**
   * @brief Typed arithmetic index mapping replacing the old View-based lookup.
   *
   * Returns the container-local index for a global element index @p ispec.
   * Valid for device and host code.
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  KOKKOS_INLINE_FUNCTION int property_index_mapping(int ispec) const {
    return ispec -
           element_ranges
               .template get<
                   specfem::tags::Tags<dimension_tag, MediumTag, PropertyTag>>()
               .begin_index();
  }

  /**
   * @brief Default constructor
   */
  value_containers_base() = default;

  /**
   * @brief Construct and fully initialise a value_containers_base.
   *
   * Constructs every slot of @c value via the supplied per-tag initializer,
   * then reads @c element_range from each slot to populate @c element_ranges.
   *
   * @tparam Initializer  Callable invocable as
   *   `initializer.template operator()<TagsType>()` returning the slot value.
   * @param  nspec_       Number of spectral elements.
   * @param  grid         GLL grid layout.
   * @param  initializer  Per-tag slot initializer.
   */
  template <typename Initializer>
  value_containers_base(int nspec_,
                        specfem::mesh_entity::element_grid<DimensionTag> grid,
                        Initializer &&initializer)
      : nspec(nspec_), element_grid(grid), value(initializer) {
    specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
      element_ranges.template get<TagsType>() =
          value.template get<TagsType>().element_range;
    });
  }

  /**
   * @brief Returns the container for a given medium and property
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  KOKKOS_INLINE_FUNCTION constexpr containers_type<dimension_tag, MediumTag,
                                                   PropertyTag> const &
  get_container() const {
    return value.template get<
        specfem::tags::Tags<dimension_tag, MediumTag, PropertyTag>>();
  }

  /**
   * @brief Copy data to host
   */
  void copy_to_host() {
    specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
      value.template get<TagsType>().copy_to_host();
    });
  }

  void copy_to_device() {
    specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
      value.template get<TagsType>().copy_to_device();
    });
  }
};

/**
 * @brief Alias mapping (dimension_tag, containers_type) to
 * value_containers_base via ContainerSets<DimensionTag>.
 */
template <
    specfem::element::dimension_tag DimensionTag,
    template <specfem::element::dimension_tag, specfem::element::medium_tag,
              specfem::element::property_tag> class containers_type>
using value_containers = value_containers_base<DimensionTag, containers_type>;

} // namespace specfem::assembly::impl
