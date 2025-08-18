#pragma once

#include "boundary_medium_container.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"

namespace specfem::assembly::boundary_values_impl {

template <specfem::element::boundary_tag BoundaryTag>
class boundary_value_container<specfem::dimension::type::dim2, BoundaryTag> {

private:
  template <specfem::dimension::type _DimensionTag,
            specfem::element::medium_tag _MediumTag>
  using _boundary_medium_container =
      boundary_medium_container<_DimensionTag, _MediumTag, BoundaryTag>;

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;
  constexpr static auto boundary_tag = BoundaryTag;

  IndexViewType property_index_mapping;
  IndexViewType::HostMirror h_property_index_mapping;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      DECLARE(((_boundary_medium_container, (_DIMENSION_TAG_, _MEDIUM_TAG_)),
               container)))

  // boundary_medium_container<DimensionTag,
  //                           specfem::element::medium_tag::acoustic,
  //                           BoundaryTag>
  //     acoustic;
  // boundary_medium_container<
  //     DimensionTag, specfem::element::medium_tag::elastic_psv, BoundaryTag>
  //     elastic;
  // boundary_medium_container<
  //     DimensionTag, specfem::element::medium_tag::poroelastic, BoundaryTag>
  //     poroelastic;

  boundary_value_container() = default;

  boundary_value_container(
      const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::boundaries<dimension_tag> &boundaries);

  void sync_to_host() {
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        CAPTURE(container) { _container_.sync_to_host(); });
  }

  void sync_to_device() {
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        CAPTURE(container) { _container_.sync_to_device(); });
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

  constexpr static auto MediumTag = AccelerationType::medium_tag;

  static_assert((BoundaryValueContainerType::dimension_tag ==
                 AccelerationType::dimension_tag),
                "DimensionTag must match AccelerationType::dimension_type");

  IndexType l_index = index;
  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE((container, boundary_value_container.container)) {
        if constexpr (MediumTag == _medium_tag_) {
          _container_.store_on_device(istep, l_index, acceleration);
        }
      });

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

  constexpr static auto MediumTag = AccelerationType::medium_tag;

  IndexType l_index = index;

  static_assert((BoundaryValueContainerType::dimension_tag ==
                 AccelerationType::dimension_tag),
                "Number of dimensions must match");

  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE((container, boundary_value_container.container)) {
        if constexpr (MediumTag == _medium_tag_) {
          _container_.load_on_device(istep, l_index, acceleration);
        }
      });

  return;
}

} // namespace specfem::assembly
