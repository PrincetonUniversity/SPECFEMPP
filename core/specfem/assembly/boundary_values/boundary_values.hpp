#pragma once

#include "dim2/impl/boundary_value_container.hpp"
#include "dim3/impl/boundary_value_container.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/tags.hpp"

namespace specfem::assembly {

template <specfem::element::dimension_tag DimensionTag> class boundary_values {
public:
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag
  boundary_values() = default;

  specfem::assembly::boundary_values_impl::boundary_value_container<
      dimension_tag, specfem::element::boundary_tag::stacey>
      stacey;

  specfem::assembly::boundary_values_impl::boundary_value_container<
      dimension_tag, specfem::element::boundary_tag::composite_stacey_dirichlet>
      composite_stacey_dirichlet;

  boundary_values(
      const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::boundaries<dimension_tag> &boundaries);

  template <specfem::element::boundary_tag BoundaryTag>
  specfem::assembly::boundary_values_impl::boundary_value_container<
      dimension_tag, BoundaryTag>
  get_container() const {
    if constexpr (BoundaryTag == specfem::element::boundary_tag::stacey) {
      return stacey;
    } else if constexpr (BoundaryTag == specfem::element::boundary_tag::
                                            composite_stacey_dirichlet) {
      return composite_stacey_dirichlet;
    } else {
      return {};
    }
  }

  void copy_to_host() {
    stacey.sync_to_host();
    composite_stacey_dirichlet.sync_to_host();
  }

  void copy_to_device() {
    stacey.sync_to_device();
    composite_stacey_dirichlet.sync_to_device();
  }
};

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

  static_assert((BoundaryValueContainerType::dimension_tag ==
                 AccelerationType::dimension_tag),
                "Number of dimensions must match");

  using TagsType = specfem::tags::Tags<dimension_tag, medium_tag>;

  IndexType l_index = index;
  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  boundary_value_container.container.template get<TagsType>().load_on_device(
      istep, l_index, acceleration);

  return;
}
} // namespace specfem::assembly
