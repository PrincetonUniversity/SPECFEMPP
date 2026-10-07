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

  const auto &mapping = container.xix.get_mapping();
  const std::size_t _index = mapping(index.ispec, index.iz, index.iy, index.ix);
  const auto mask = index.template get_mask<simd>();

  if constexpr (on_device) {
    point.xix() = Kokkos::Experimental::simd_partial_load(
        &container.xix[_index], mask, tag_type());
    point.xiy() = Kokkos::Experimental::simd_partial_load(
        &container.xiy[_index], mask, tag_type());
    point.xiz() = Kokkos::Experimental::simd_partial_load(
        &container.xiz[_index], mask, tag_type());
    point.etax() = Kokkos::Experimental::simd_partial_load(
        &container.etax[_index], mask, tag_type());
    point.etay() = Kokkos::Experimental::simd_partial_load(
        &container.etay[_index], mask, tag_type());
    point.etaz() = Kokkos::Experimental::simd_partial_load(
        &container.etaz[_index], mask, tag_type());
    point.gammax() = Kokkos::Experimental::simd_partial_load(
        &container.gammax[_index], mask, tag_type());
    point.gammay() = Kokkos::Experimental::simd_partial_load(
        &container.gammay[_index], mask, tag_type());
    point.gammaz() = Kokkos::Experimental::simd_partial_load(
        &container.gammaz[_index], mask, tag_type());
    if constexpr (load_jacobian) {
      point.jacobian() = Kokkos::Experimental::simd_partial_load(
          &container.jacobian[_index], mask, tag_type());
    }
  } else {
    point.xix() = Kokkos::Experimental::simd_partial_load(
        &container.h_xix[_index], mask, tag_type());
    point.xiy() = Kokkos::Experimental::simd_partial_load(
        &container.h_xiy[_index], mask, tag_type());
    point.xiz() = Kokkos::Experimental::simd_partial_load(
        &container.h_xiz[_index], mask, tag_type());
    point.etax() = Kokkos::Experimental::simd_partial_load(
        &container.h_etax[_index], mask, tag_type());
    point.etay() = Kokkos::Experimental::simd_partial_load(
        &container.h_etay[_index], mask, tag_type());
    point.etaz() = Kokkos::Experimental::simd_partial_load(
        &container.h_etaz[_index], mask, tag_type());
    point.gammax() = Kokkos::Experimental::simd_partial_load(
        &container.h_gammax[_index], mask, tag_type());
    point.gammay() = Kokkos::Experimental::simd_partial_load(
        &container.h_gammay[_index], mask, tag_type());
    point.gammaz() = Kokkos::Experimental::simd_partial_load(
        &container.h_gammaz[_index], mask, tag_type());
    if constexpr (load_jacobian) {
      point.jacobian() = Kokkos::Experimental::simd_partial_load(
          &container.h_jacobian[_index], mask, tag_type());
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

  const auto &mapping = container.xix.get_mapping();
  const std::size_t _index = mapping(index.ispec, index.iz, index.iy, index.ix);

  if constexpr (on_device) {
    point.xix() = container.xix.get_base_view().data()[_index];
    point.xiy() = container.xiy.get_base_view().data()[_index];
    point.xiz() = container.xiz.get_base_view().data()[_index];
    point.etax() = container.etax.get_base_view().data()[_index];
    point.etay() = container.etay.get_base_view().data()[_index];
    point.etaz() = container.etaz.get_base_view().data()[_index];
    point.gammax() = container.gammax.get_base_view().data()[_index];
    point.gammay() = container.gammay.get_base_view().data()[_index];
    point.gammaz() = container.gammaz.get_base_view().data()[_index];
    if constexpr (load_jacobian) {
      point.jacobian() = container.jacobian.get_base_view().data()[_index];
    }
  } else {
    point.xix() = container.h_xix[_index];
    point.xiy() = container.h_xiy[_index];
    point.xiz() = container.h_xiz[_index];
    point.etax() = container.h_etax[_index];
    point.etay() = container.h_etay[_index];
    point.etaz() = container.h_etaz[_index];
    point.gammax() = container.h_gammax[_index];
    point.gammay() = container.h_gammay[_index];
    point.gammaz() = container.h_gammaz[_index];
    if constexpr (load_jacobian) {
      point.jacobian() = container.h_jacobian[_index];
    }
  }
}

} // namespace specfem::assembly
