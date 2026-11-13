#pragma once

#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @brief Internally load a 2D SIMD Jacobian matrix accessor.
 *
 * This is an internal implementation detail and is not part of the public API.
 */
template <
    bool on_device, typename IndexType, typename ContainerType,
    typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            IndexType::dimension_tag == specfem::element::dimension_tag::dim2 &&
            IndexType::using_simd && PointType::simd::using_simd &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                           const ContainerType &container,
                                           PointType &point) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  using simd = typename PointType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool StoreJacobian = PointType::store_jacobian;

  const auto &mapping = container.xix.mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  const auto mask = index.template get_mask<simd>();

  if constexpr (on_device) {
    point.xix() = Kokkos::Experimental::simd_partial_load(
        &container.xix.data()[_index], mask, tag_type());
    point.gammax() = Kokkos::Experimental::simd_partial_load(
        &container.gammax.data()[_index], mask, tag_type());
    point.xiz() = Kokkos::Experimental::simd_partial_load(
        &container.xiz.data()[_index], mask, tag_type());
    point.gammaz() = Kokkos::Experimental::simd_partial_load(
        &container.gammaz.data()[_index], mask, tag_type());
    if constexpr (StoreJacobian) {
      point.jacobian() = Kokkos::Experimental::simd_partial_load(
          &container.jacobian.data()[_index], mask, tag_type());
    }
  } else {
    point.xix() = Kokkos::Experimental::simd_partial_load(
        &container.h_xix.data()[_index], mask, tag_type());
    point.gammax() = Kokkos::Experimental::simd_partial_load(
        &container.h_gammax.data()[_index], mask, tag_type());
    point.xiz() = Kokkos::Experimental::simd_partial_load(
        &container.h_xiz.data()[_index], mask, tag_type());
    point.gammaz() = Kokkos::Experimental::simd_partial_load(
        &container.h_gammaz.data()[_index], mask, tag_type());
    if constexpr (StoreJacobian) {
      point.jacobian() = Kokkos::Experimental::simd_partial_load(
          &container.h_jacobian.data()[_index], mask, tag_type());
    }
  }
}

/**
 * @brief Internally load a 2D scalar Jacobian matrix accessor.
 *
 * This is an internal implementation detail and is not part of the public API.
 */
template <
    bool on_device, typename IndexType, typename ContainerType,
    typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            IndexType::dimension_tag == specfem::element::dimension_tag::dim2 &&
            !IndexType::using_simd && !PointType::simd::using_simd &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                           const ContainerType &container,
                                           PointType &point) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PointType::store_jacobian;

  const auto &mapping = container.xix.mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  if constexpr (on_device) {
    point.xix() = container.xix.data()[_index];
    point.gammax() = container.gammax.data()[_index];
    point.xiz() = container.xiz.data()[_index];
    point.gammaz() = container.gammaz.data()[_index];
    if constexpr (StoreJacobian) {
      point.jacobian() = container.jacobian.data()[_index];
    }
  } else {
    point.xix() = container.h_xix.data()[_index];
    point.gammax() = container.h_gammax.data()[_index];
    point.xiz() = container.h_xiz.data()[_index];
    point.gammaz() = container.h_gammaz.data()[_index];
    if constexpr (StoreJacobian) {
      point.jacobian() = container.h_jacobian.data()[_index];
    }
  }
}

} // namespace specfem::assembly
