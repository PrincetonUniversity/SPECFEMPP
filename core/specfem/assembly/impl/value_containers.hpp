#pragma once

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

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  int nspec; ///< Total number of spectral elements
  specfem::mesh_entity::element_grid<DimensionTag> element_grid; ///< GLL grid
                                                                 ///< layout

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto combinations = ContainerSetsType::combinations;

  IndexViewType property_index_mapping; ///< View to store property index
                                        ///< mapping
  IndexViewType::HostMirror h_property_index_mapping; ///< Host mirror of
                                                      ///< property index
                                                      ///< mapping

  template <bool on_device>
  KOKKOS_INLINE_FUNCTION constexpr
      typename std::conditional<on_device, IndexViewType,
                                IndexViewType::HostMirror>::type
      get_property_index_mapping() const {
    if constexpr (on_device) {
      return property_index_mapping;
    } else {
      return h_property_index_mapping;
    }
  }

  template <typename TagsType>
  using ContainerType =
      containers_type<TagsType::dimension_tag, TagsType::medium_tag,
                      TagsType::property_tag>;

  specfem::tag_dispatch::TypedStorage<ContainerType, decltype(combinations)>
      value;

  /**
   * @brief Default constructor
   */
  value_containers_base() = default;

  /**
   * @brief Construct and fully initialise a value_containers_base.
   *
   * Allocates @c property_index_mapping and its host mirror (filled with -1),
   * constructs every slot of @c value via the supplied initializer factory,
   * and deep-copies the index mapping to device.
   *
   * @tparam InitializerFactory  Callable `(HostMirrorView) -> PerTagFunctor`.
   *   The returned @c PerTagFunctor must be invocable as
   *   `functor.template operator()<TagsType>()` returning the slot value.
   * @param  nspec_           Number of spectral elements.
   * @param  grid             GLL grid layout.
   * @param  label            Kokkos label for the property index mapping view.
   * @param  make_initializer Factory called once with @c
   * h_property_index_mapping to produce the per-tag slot initializer.
   */
  template <typename InitializerFactory>
  value_containers_base(int nspec_,
                        specfem::mesh_entity::element_grid<DimensionTag> grid,
                        const std::string &label,
                        InitializerFactory &&make_initializer)
      : nspec(nspec_), element_grid(grid),
        property_index_mapping(label, nspec_),
        h_property_index_mapping([](const auto &pm) {
          auto h = Kokkos::create_mirror_view(pm);
          Kokkos::deep_copy(h, -1);
          return h;
        }(property_index_mapping)),
        value(make_initializer(h_property_index_mapping)) {
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  }

  /**
   * @brief Returns the container for a given medium and property
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  KOKKOS_INLINE_FUNCTION
      constexpr containers_type<dimension_tag, MediumTag, PropertyTag> const &
      get_container() const {
    return value.template get<
        specfem::tags::Tags<dimension_tag, MediumTag, PropertyTag> >();
  }

  /**
   * @brief Copy data to host
   */
  void copy_to_host() {
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
      value.template get<TagsType>().copy_to_host();
    });
  }

  void copy_to_device() {
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
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
