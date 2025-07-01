#pragma once

#include "boundary_medium_container.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"

namespace specfem::assembly {

template <specfem::dimension::type DimensionTag,
          specfem::element::boundary_tag BoundaryTag>
class boundary_value_container {

public:
  constexpr static auto dimension = DimensionTag;
  constexpr static auto boundary_tag = BoundaryTag;

  specfem::kokkos::DeviceView1d<int> property_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_property_index_mapping;

  specfem::assembly::impl::boundary_medium_container<
      DimensionTag, specfem::element::medium_tag::acoustic, BoundaryTag>
      acoustic;
  specfem::assembly::impl::boundary_medium_container<
      DimensionTag, specfem::element::medium_tag::elastic_psv, BoundaryTag>
      elastic;
  specfem::assembly::impl::boundary_medium_container<
      DimensionTag, specfem::element::medium_tag::poroelastic, BoundaryTag>
      poroelastic;

  boundary_value_container() = default;

  boundary_value_container(const int nstep, const specfem::assembly::mesh mesh,
                           const specfem::assembly::element_types element_types,
                           const specfem::assembly::boundaries boundaries);

  void sync_to_host() {
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    acoustic.sync_to_host();
    elastic.sync_to_host();
  }

  void sync_to_device() {
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
    acoustic.sync_to_device();
    elastic.sync_to_device();
    poroelastic.sync_to_device();
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

  static_assert((BoundaryValueContainerType::dimension ==
                 AccelerationType::dimension_tag),
                "DimensionTag must match AccelerationType::dimension_type");

  IndexType l_index = index;
  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  if constexpr (MediumTag == specfem::element::medium_tag::acoustic) {
    boundary_value_container.acoustic.store_on_device(istep, l_index,
                                                      acceleration);
  } else if constexpr (MediumTag == specfem::element::medium_tag::elastic_psv) {
    boundary_value_container.elastic.store_on_device(istep, l_index,
                                                     acceleration);
  } else if constexpr (MediumTag == specfem::element::medium_tag::poroelastic) {
    boundary_value_container.poroelastic.store_on_device(istep, l_index,
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

  constexpr static auto MediumTag = AccelerationType::medium_tag;

  IndexType l_index = index;

  static_assert((BoundaryValueContainerType::dimension ==
                 AccelerationType::dimension_tag),
                "Number of dimensions must match");

  l_index.ispec = boundary_value_container.property_index_mapping(index.ispec);

  if constexpr (MediumTag == specfem::element::medium_tag::acoustic) {
    boundary_value_container.acoustic.load_on_device(istep, l_index,
                                                     acceleration);
  } else if constexpr (MediumTag == specfem::element::medium_tag::elastic_psv) {
    boundary_value_container.elastic.load_on_device(istep, l_index,
                                                    acceleration);
  } else if constexpr (MediumTag == specfem::element::medium_tag::poroelastic) {
    boundary_value_container.poroelastic.load_on_device(istep, l_index,
                                                        acceleration);
  }

  return;
}

} // namespace specfem::assembly
