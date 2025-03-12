#ifndef _COMPUTE_BOUNDARIES_VALUES_BOUNDARY_VALUES_CONTAINER_HPP
#define _COMPUTE_BOUNDARIES_VALUES_BOUNDARY_VALUES_CONTAINER_HPP

#include "compute/boundaries/boundaries.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "impl/boundary_medium_container.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace compute {

template <specfem::dimension::type DimensionType,
          specfem::element::boundary_tag BoundaryTag>
class boundary_value_container {

public:
  constexpr static auto dimension = DimensionType;
  constexpr static auto boundary_tag = BoundaryTag;

  specfem::kokkos::DeviceView1d<int> property_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_property_index_mapping;

  specfem::compute::impl::boundary_medium_container<
      DimensionType, specfem::element::medium_tag::acoustic, BoundaryTag>
      acoustic;
  specfem::compute::impl::boundary_medium_container<
      DimensionType, specfem::element::medium_tag::elastic_sv, BoundaryTag>
      elastic;

  boundary_value_container() = default;

  boundary_value_container(const int nstep, const specfem::compute::mesh mesh,
                           const specfem::compute::element_types element_types,
                           const specfem::compute::boundaries boundaries);

  void sync_to_host() {
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    acoustic.sync_to_host();
    elastic.sync_to_host();
  }

  void sync_to_device() {
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
    acoustic.sync_to_device();
    elastic.sync_to_device();
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

  constexpr static auto MediumTag = AccelerationType::medium_tag;

  static_assert(
      (BoundaryValueContainerType::dimension == AccelerationType::dimension),
      "DimensionType must match AccelerationType::dimension_type");

  IndexType l_index = index;
  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  if constexpr (MediumTag == specfem::element::medium_tag::acoustic) {
    boundary_value_container.acoustic.store_on_device(istep, l_index,
                                                      acceleration);
  } else if constexpr (MediumTag == specfem::element::medium_tag::elastic_sv) {
    boundary_value_container.elastic.store_on_device(istep, l_index,
                                                     acceleration);
  }

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

  constexpr static auto MediumType = AccelerationType::medium_tag;

  IndexType l_index = index;

  static_assert(
      (BoundaryValueContainerType::dimension == AccelerationType::dimension),
      "Number of dimensions must match");

  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
    boundary_value_container.acoustic.load_on_device(istep, l_index,
                                                     acceleration);
  } else if constexpr (MediumType == specfem::element::medium_tag::elastic_sv) {
    boundary_value_container.elastic.load_on_device(istep, l_index,
                                                    acceleration);
  }

  return;
}

} // namespace compute
} // namespace specfem

#endif
