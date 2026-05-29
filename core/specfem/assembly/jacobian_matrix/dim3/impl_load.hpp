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
KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                           const ContainerType &container,
                                           PointType &point) {

  using simd = typename PointType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool load_jacobian = PointType::store_jacobian;

  const auto mask = index.template get_mask<simd>();

  if constexpr (on_device) {
    point.xix = Kokkos::Experimental::simd_partial_load(
        &container.xix(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.xiy = Kokkos::Experimental::simd_partial_load(
        &container.xiy(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.xiz = Kokkos::Experimental::simd_partial_load(
        &container.xiz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.etax = Kokkos::Experimental::simd_partial_load(
        &container.etax(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.etay = Kokkos::Experimental::simd_partial_load(
        &container.etay(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.etaz = Kokkos::Experimental::simd_partial_load(
        &container.etaz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.gammax = Kokkos::Experimental::simd_partial_load(
        &container.gammax(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.gammay = Kokkos::Experimental::simd_partial_load(
        &container.gammay(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.gammaz = Kokkos::Experimental::simd_partial_load(
        &container.gammaz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    if constexpr (load_jacobian) {
      point.jacobian = Kokkos::Experimental::simd_partial_load(
          &container.jacobian(index.ispec, index.iz, index.iy, index.ix), mask,
          tag_type());
    }
  } else {
    point.xix = Kokkos::Experimental::simd_partial_load(
        &container.h_xix(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.xiy = Kokkos::Experimental::simd_partial_load(
        &container.h_xiy(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.xiz = Kokkos::Experimental::simd_partial_load(
        &container.h_xiz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.etax = Kokkos::Experimental::simd_partial_load(
        &container.h_etax(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.etay = Kokkos::Experimental::simd_partial_load(
        &container.h_etay(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.etaz = Kokkos::Experimental::simd_partial_load(
        &container.h_etaz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.gammax = Kokkos::Experimental::simd_partial_load(
        &container.h_gammax(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.gammay = Kokkos::Experimental::simd_partial_load(
        &container.h_gammay(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    point.gammaz = Kokkos::Experimental::simd_partial_load(
        &container.h_gammaz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    if constexpr (load_jacobian) {
      point.jacobian = Kokkos::Experimental::simd_partial_load(
          &container.h_jacobian(index.ispec, index.iz, index.iy, index.ix),
          mask, tag_type());
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
KOKKOS_FORCEINLINE_FUNCTION void impl_load(const PointIndexType &index,
                                           const ContainerType &container,
                                           PointType &point) {

  constexpr static bool load_jacobian = PointType::store_jacobian;

  if constexpr (on_device) {
    point.xix = container.xix(index.ispec, index.iz, index.iy, index.ix);
    point.xiy = container.xiy(index.ispec, index.iz, index.iy, index.ix);
    point.xiz = container.xiz(index.ispec, index.iz, index.iy, index.ix);
    point.etax = container.etax(index.ispec, index.iz, index.iy, index.ix);
    point.etay = container.etay(index.ispec, index.iz, index.iy, index.ix);
    point.etaz = container.etaz(index.ispec, index.iz, index.iy, index.ix);
    point.gammax = container.gammax(index.ispec, index.iz, index.iy, index.ix);
    point.gammay = container.gammay(index.ispec, index.iz, index.iy, index.ix);
    point.gammaz = container.gammaz(index.ispec, index.iz, index.iy, index.ix);
    if constexpr (load_jacobian) {
      point.jacobian =
          container.jacobian(index.ispec, index.iz, index.iy, index.ix);
    }
  } else {
    point.xix = container.h_xix(index.ispec, index.iz, index.iy, index.ix);
    point.xiy = container.h_xiy(index.ispec, index.iz, index.iy, index.ix);
    point.xiz = container.h_xiz(index.ispec, index.iz, index.iy, index.ix);
    point.etax = container.h_etax(index.ispec, index.iz, index.iy, index.ix);
    point.etay = container.h_etay(index.ispec, index.iz, index.iy, index.ix);
    point.etaz = container.h_etaz(index.ispec, index.iz, index.iy, index.ix);
    point.gammax =
        container.h_gammax(index.ispec, index.iz, index.iy, index.ix);
    point.gammay =
        container.h_gammay(index.ispec, index.iz, index.iy, index.ix);
    point.gammaz =
        container.h_gammaz(index.ispec, index.iz, index.iy, index.ix);
    if constexpr (load_jacobian) {
      point.jacobian =
          container.h_jacobian(index.ispec, index.iz, index.iy, index.ix);
    }
  }
}

} // namespace specfem::assembly
