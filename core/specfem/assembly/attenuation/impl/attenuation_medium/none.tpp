#pragma once

#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/enums.hpp"

namespace specfem::assembly::impl {

/**
 * @brief Empty specialization of attenuation_medium for attenuation_tag::none.
 *
 * This struct has no data members and all operations are no-ops, ensuring zero
 * overhead for element combinations that do not use attenuation.
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct attenuation_medium<DimensionTag, MediumTag, PropertyTag,
                          specfem::element::attenuation_tag::none> {
  attenuation_medium() = default;
  void copy_to_host() {}
  void copy_to_device() {}

  template <typename IndexType, typename PointType>
  KOKKOS_INLINE_FUNCTION void load_device_values(const IndexType &,
                                                 PointType &) const {}

  template <typename IndexType, typename PointType>
  KOKKOS_INLINE_FUNCTION void
  store_device_values(const IndexType &, const PointType &) const {}
};

} // namespace specfem::assembly::impl
