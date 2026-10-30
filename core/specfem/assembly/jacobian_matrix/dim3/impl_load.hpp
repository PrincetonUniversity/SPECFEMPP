#pragma once

#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @brief Internally load a 3D SIMD Jacobian matrix accessor.
 *
 * This is an internal implementation detail and is not part of the public API.
 */
template <
    bool on_device, typename IndexType, typename ContainerType,
    typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            IndexType::dimension_tag == specfem::element::dimension_tag::dim3 &&
            IndexType::using_simd && PointType::simd::using_simd &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                           const ContainerType &container,
                                           PointType &point) {

  using simd = typename PointType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool load_jacobian = PointType::store_jacobian;

  const auto &mapping = container.xix.mapping();
  const std::size_t _index = mapping(index.ispec, index.iz, index.iy, index.ix);
  const auto mask = index.template get_mask<simd>();

  if constexpr (on_device) {
    point.xix() = Kokkos::Experimental::simd_partial_load(
        &container.xix.data()[_index], mask, tag_type());
    point.xiy() = Kokkos::Experimental::simd_partial_load(
        &container.xiy.data()[_index], mask, tag_type());
    point.xiz() = Kokkos::Experimental::simd_partial_load(
        &container.xiz.data()[_index], mask, tag_type());
    point.etax() = Kokkos::Experimental::simd_partial_load(
        &container.etax.data()[_index], mask, tag_type());
    point.etay() = Kokkos::Experimental::simd_partial_load(
        &container.etay.data()[_index], mask, tag_type());
    point.etaz() = Kokkos::Experimental::simd_partial_load(
        &container.etaz.data()[_index], mask, tag_type());
    point.gammax() = Kokkos::Experimental::simd_partial_load(
        &container.gammax.data()[_index], mask, tag_type());
    point.gammay() = Kokkos::Experimental::simd_partial_load(
        &container.gammay.data()[_index], mask, tag_type());
    point.gammaz() = Kokkos::Experimental::simd_partial_load(
        &container.gammaz.data()[_index], mask, tag_type());
    if constexpr (load_jacobian) {
      point.jacobian() = Kokkos::Experimental::simd_partial_load(
          &container.jacobian.data()[_index], mask, tag_type());
    }
  } else {
    point.xix() = Kokkos::Experimental::simd_partial_load(
        &container.h_xix.data()[_index], mask, tag_type());
    point.xiy() = Kokkos::Experimental::simd_partial_load(
        &container.h_xiy.data()[_index], mask, tag_type());
    point.xiz() = Kokkos::Experimental::simd_partial_load(
        &container.h_xiz.data()[_index], mask, tag_type());
    point.etax() = Kokkos::Experimental::simd_partial_load(
        &container.h_etax.data()[_index], mask, tag_type());
    point.etay() = Kokkos::Experimental::simd_partial_load(
        &container.h_etay.data()[_index], mask, tag_type());
    point.etaz() = Kokkos::Experimental::simd_partial_load(
        &container.h_etaz.data()[_index], mask, tag_type());
    point.gammax() = Kokkos::Experimental::simd_partial_load(
        &container.h_gammax.data()[_index], mask, tag_type());
    point.gammay() = Kokkos::Experimental::simd_partial_load(
        &container.h_gammay.data()[_index], mask, tag_type());
    point.gammaz() = Kokkos::Experimental::simd_partial_load(
        &container.h_gammaz.data()[_index], mask, tag_type());
    if constexpr (load_jacobian) {
      point.jacobian() = Kokkos::Experimental::simd_partial_load(
          &container.h_jacobian.data()[_index], mask, tag_type());
    }
  }
}

/**
 * @brief Internally load a 3D scalar Jacobian matrix accessor.
 *
 * This is an internal implementation detail and is not part of the public API.
 */
template <
    bool on_device, typename PointIndexType, typename ContainerType,
    typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<PointIndexType>::value &&
            PointIndexType::dimension_tag ==
                specfem::element::dimension_tag::dim3 &&
            !PointIndexType::using_simd && !PointType::simd::using_simd &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(const PointIndexType &index,
                                           const ContainerType &container,
                                           PointType &point) {

  constexpr static bool load_jacobian = PointType::store_jacobian;

  const auto &mapping = container.xix.mapping();
  const std::size_t _index = mapping(index.ispec, index.iz, index.iy, index.ix);

  if constexpr (on_device) {
    point.xix() = container.xix.data()[_index];
    point.xiy() = container.xiy.data()[_index];
    point.xiz() = container.xiz.data()[_index];
    point.etax() = container.etax.data()[_index];
    point.etay() = container.etay.data()[_index];
    point.etaz() = container.etaz.data()[_index];
    point.gammax() = container.gammax.data()[_index];
    point.gammay() = container.gammay.data()[_index];
    point.gammaz() = container.gammaz.data()[_index];
    if constexpr (load_jacobian) {
      point.jacobian() = container.jacobian.data()[_index];
    }
  } else {
    point.xix() = container.h_xix.data()[_index];
    point.xiy() = container.h_xiy.data()[_index];
    point.xiz() = container.h_xiz.data()[_index];
    point.etax() = container.h_etax.data()[_index];
    point.etay() = container.h_etay.data()[_index];
    point.etaz() = container.h_etaz.data()[_index];
    point.gammax() = container.h_gammax.data()[_index];
    point.gammay() = container.h_gammay.data()[_index];
    point.gammaz() = container.h_gammaz.data()[_index];
    if constexpr (load_jacobian) {
      point.jacobian() = container.h_jacobian.data()[_index];
    }
  }
}

} // namespace specfem::assembly
