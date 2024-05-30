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
  specfem::kokkos::DeviceView1d<int> property_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_property_index_mapping;

  specfem::compute::impl::boundary_medium_container<
      DimensionType, specfem::element::medium_tag::acoustic, BoundaryTag>
      acoustic;
  specfem::compute::impl::boundary_medium_container<
      DimensionType, specfem::element::medium_tag::elastic, BoundaryTag>
      elastic;

  boundary_value_container() = default;

  boundary_value_container(const int nstep, const specfem::compute::mesh mesh,
                           const specfem::compute::properties properties,
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

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType,
          specfem::element::boundary_tag BoundaryTag>
KOKKOS_FUNCTION void load_on_device(
    const int istep, const specfem::point::index index,
    const specfem::compute::boundary_value_container<DimensionType, BoundaryTag>
        &boundary_value_container,
    specfem::point::field<DimensionType, MediumType, false, false, true, false>
        &acceleration) {

  const int ispec =
      boundary_value_container.property_index_mapping(index.ispec);
  const int iz = index.iz;
  const int ix = index.ix;

  if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
    boundary_value_container.acoustic.load_on_device(istep, ispec, iz, ix,
                                                     acceleration);
  } else if constexpr (MediumType == specfem::element::medium_tag::elastic) {
    boundary_value_container.elastic.load_on_device(istep, ispec, iz, ix,
                                                    acceleration);
  }

  return;
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::boundary_tag BoundaryTag>
KOKKOS_FUNCTION void store_on_device(
    const int istep, const specfem::point::index index,
    const specfem::point::field<DimensionType, MediumTag, false, false, true,
                                false> &acceleration,
    const specfem::compute::boundary_value_container<DimensionType, BoundaryTag>
        &boundary_value_container) {

  const int ispec =
      boundary_value_container.property_index_mapping(index.ispec);
  const int iz = index.iz;
  const int ix = index.ix;

  if constexpr (MediumTag == specfem::element::medium_tag::acoustic) {
    boundary_value_container.acoustic.store_on_device(istep, ispec, iz, ix,
                                                      acceleration);
  } else if constexpr (MediumTag == specfem::element::medium_tag::elastic) {
    boundary_value_container.elastic.store_on_device(istep, ispec, iz, ix,
                                                     acceleration);
  }

  return;
}

} // namespace compute
} // namespace specfem

#endif
