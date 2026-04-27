#pragma once

#include "boundary_medium_container.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"

namespace specfem::assembly::boundary_values_impl {

template <specfem::element::boundary_tag BoundaryTag>
class boundary_value_container<specfem::element::dimension_tag::dim3,
                               BoundaryTag> {

private:
  template <typename TagsType>
  using BoundaryMediumTemplateType =
      boundary_medium_container<TagsType::dimension_tag, TagsType::medium_tag,
                                BoundaryTag>;

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr static auto boundary_tag = BoundaryTag;

  static constexpr auto combinations_by_medium =
      DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic);

  IndexViewType property_index_mapping;
  IndexViewType::HostMirror h_property_index_mapping;

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
