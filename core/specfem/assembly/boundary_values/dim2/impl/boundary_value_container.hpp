#pragma once

#include "boundary_medium_container.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"

namespace specfem::assembly::boundary_values_impl {

template <specfem::element::boundary_tag BoundaryTag>
class boundary_value_container<specfem::element::dimension_tag::dim2,
                               BoundaryTag> {

private:
  template <typename TagsType>
  using BoundaryMediumTemplateType =
      boundary_medium_container<TagsType::dimension_tag, TagsType::medium_tag,
                                BoundaryTag>;

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;
  constexpr static auto boundary_tag = BoundaryTag;

  static constexpr auto combinations_by_medium =
      DIMENSION_SET(dim2) *
      MEDIUM_SET(elastic_psv, elastic_psv_t, elastic_sh, acoustic, poroelastic);

  IndexViewType property_index_mapping;
  IndexViewType::host_mirror_type h_property_index_mapping;

  specfem::tag_dispatch::TypedStorage<BoundaryMediumTemplateType,
                                      decltype(combinations_by_medium)>
      container;

  boundary_value_container() = default;

  boundary_value_container(
      const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::boundaries<dimension_tag> &boundaries);

  void sync_to_host() {
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    specfem::tag_dispatch::for_each(
        combinations_by_medium, [&]<typename TagsType>() {
          container.template get<TagsType>().sync_to_host();
        });
  }

  void sync_to_device() {
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
    specfem::tag_dispatch::for_each(
        combinations_by_medium, [&]<typename TagsType>() {
          container.template get<TagsType>().sync_to_device();
        });
  }
};
} // namespace specfem::assembly::boundary_values_impl

namespace specfem::assembly {

template <typename IndexType, typename AccelerationType,
          typename BoundaryValueContainerType,
          typename std::enable_if_t<
              ((BoundaryValueContainerType::boundary_tag ==
                specfem::element::boundary_tag::none) ||
               (BoundaryValueContainerType::boundary_tag ==
                specfem::element::boundary_tag::acoustic_free_surface)),
              int> = 0>
KOKKOS_INLINE_FUNCTION void
store_on_device(const int istep, const IndexType index,
                const AccelerationType &acceleration,
                const BoundaryValueContainerType &boundary_value_container) {
  return;
}

template <typename IndexType, typename AccelerationType,
          typename BoundaryValueContainerType,
          typename std::enable_if_t<
              ((BoundaryValueContainerType::boundary_tag ==
                specfem::element::boundary_tag::stacey) ||
               (BoundaryValueContainerType::boundary_tag ==
                specfem::element::boundary_tag::composite_stacey_dirichlet)),
              int> = 0>
KOKKOS_FUNCTION void
store_on_device(const int istep, const IndexType index,
                const AccelerationType &acceleration,
                const BoundaryValueContainerType &boundary_value_container) {

  if (boundary_value_container.property_index_mapping.size() == 0)
    return;

  constexpr static auto medium_tag = AccelerationType::medium_tag;
  constexpr static auto dimension_tag = AccelerationType::dimension_tag;

  static_assert((BoundaryValueContainerType::dimension_tag ==
                 AccelerationType::dimension_tag),
                "DimensionTag must match AccelerationType::dimension_type");

  using TagsType = specfem::tags::Tags<dimension_tag, medium_tag>;

  IndexType l_index = index;
  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  boundary_value_container.container.template get<TagsType>().store_on_device(
      istep, l_index, acceleration);

  return;
}

template <typename IndexType, typename AccelerationType,
          typename BoundaryValueContainerType,
          typename std::enable_if_t<
              ((BoundaryValueContainerType::boundary_tag ==
                specfem::element::boundary_tag::stacey) ||
               (BoundaryValueContainerType::boundary_tag ==
                specfem::element::boundary_tag::composite_stacey_dirichlet)),
              int> = 0>
KOKKOS_FUNCTION void
load_on_device(const int istep, const IndexType index,
               const BoundaryValueContainerType &boundary_value_container,
               AccelerationType &acceleration) {

  if (boundary_value_container.property_index_mapping.size() == 0)
    return;

  constexpr static auto medium_tag = AccelerationType::medium_tag;
  constexpr static auto dimension_tag = AccelerationType::dimension_tag;

  IndexType l_index = index;

  static_assert((BoundaryValueContainerType::dimension_tag ==
                 AccelerationType::dimension_tag),
                "Number of dimensions must match");

  using TagsType = specfem::tags::Tags<dimension_tag, medium_tag>;

  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  boundary_value_container.container.template get<TagsType>().load_on_device(
      istep, l_index, acceleration);

  return;
}

} // namespace specfem::assembly
