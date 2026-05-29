#pragma once

#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

template <
    bool on_device, typename IndexType, typename ContainerType,
    typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            IndexType::dimension_tag == specfem::element::dimension_tag::dim3 &&
            IndexType::using_simd && PointType::simd::using_simd &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
inline void impl_store(const IndexType &index, const ContainerType &container,
                       const PointType &point) {

  using simd = typename PointType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool store_jacobian = PointType::store_jacobian;

  const auto &mapping = container.xix.get_mapping();
  const std::size_t _index = mapping(index.ispec, index.iz, index.iy, index.ix);
  const auto mask = index.template get_mask<simd>();

  if constexpr (on_device) {
    Kokkos::Experimental::simd_partial_store(point.xix, &container.xix[_index],
                                             mask, tag_type());
    Kokkos::Experimental::simd_partial_store(point.xiy, &container.xiy[_index],
                                             mask, tag_type());
    Kokkos::Experimental::simd_partial_store(point.xiz, &container.xiz[_index],
                                             mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etax, &container.etax[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etay, &container.etay[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etaz, &container.etaz[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammax, &container.gammax[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammay, &container.gammay[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammaz, &container.gammaz[_index], mask, tag_type());
    if constexpr (store_jacobian) {
      Kokkos::Experimental::simd_partial_store(
          point.jacobian, &container.jacobian[_index], mask, tag_type());
    }
  } else {
    Kokkos::Experimental::simd_partial_store(
        point.xix, &container.h_xix[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.xiy, &container.h_xiy[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.xiz, &container.h_xiz[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etax, &container.h_etax[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etay, &container.h_etay[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etaz, &container.h_etaz[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammax, &container.h_gammax[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammay, &container.h_gammay[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammaz, &container.h_gammaz[_index], mask, tag_type());
    if constexpr (store_jacobian) {
      Kokkos::Experimental::simd_partial_store(
          point.jacobian, &container.h_jacobian[_index], mask, tag_type());
    }
  }
}

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
KOKKOS_FORCEINLINE_FUNCTION void impl_store(const PointIndexType &index,
                                            const ContainerType &container,
                                            const PointType &point) {

  constexpr static bool store_jacobian = PointType::store_jacobian;

  const auto &mapping = container.xix.get_mapping();
  const std::size_t _index = mapping(index.ispec, index.iz, index.iy, index.ix);

  if constexpr (on_device) {
    container.xix.get_base_view().data()[_index] = point.xix;
    container.xiy.get_base_view().data()[_index] = point.xiy;
    container.xiz.get_base_view().data()[_index] = point.xiz;
    container.etax.get_base_view().data()[_index] = point.etax;
    container.etay.get_base_view().data()[_index] = point.etay;
    container.etaz.get_base_view().data()[_index] = point.etaz;
    container.gammax.get_base_view().data()[_index] = point.gammax;
    container.gammay.get_base_view().data()[_index] = point.gammay;
    container.gammaz.get_base_view().data()[_index] = point.gammaz;
    if constexpr (store_jacobian) {
      container.jacobian.get_base_view().data()[_index] = point.jacobian;
    }
  } else {
    container.h_xix[_index] = point.xix;
    container.h_xiy[_index] = point.xiy;
    container.h_xiz[_index] = point.xiz;
    container.h_etax[_index] = point.etax;
    container.h_etay[_index] = point.etay;
    container.h_etaz[_index] = point.etaz;
    container.h_gammax[_index] = point.gammax;
    container.h_gammay[_index] = point.gammay;
    container.h_gammaz[_index] = point.gammaz;
    if constexpr (store_jacobian) {
      container.h_jacobian[_index] = point.jacobian;
    }
  }
}

} // namespace specfem::assembly
