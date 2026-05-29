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

  const auto mask = index.template get_mask<simd>();

  if constexpr (on_device) {
    Kokkos::Experimental::simd_partial_store(
        point.xix, &container.xix(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.xiy, &container.xiy(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.xiz, &container.xiz(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etax, &container.etax(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etay, &container.etay(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etaz, &container.etaz(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammax,
        &container.gammax(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammay,
        &container.gammay(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammaz,
        &container.gammaz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    if constexpr (store_jacobian) {
      Kokkos::Experimental::simd_partial_store(
          point.jacobian,
          &container.jacobian(index.ispec, index.iz, index.iy, index.ix), mask,
          tag_type());
    }
  } else {
    Kokkos::Experimental::simd_partial_store(
        point.xix, &container.h_xix(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.xiy, &container.h_xiy(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.xiz, &container.h_xiz(index.ispec, index.iz, index.iy, index.ix),
        mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etax,
        &container.h_etax(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etay,
        &container.h_etay(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.etaz,
        &container.h_etaz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammax,
        &container.h_gammax(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammay,
        &container.h_gammay(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    Kokkos::Experimental::simd_partial_store(
        point.gammaz,
        &container.h_gammaz(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
    if constexpr (store_jacobian) {
      Kokkos::Experimental::simd_partial_store(
          point.jacobian,
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
KOKKOS_FORCEINLINE_FUNCTION void impl_store(const PointIndexType &index,
                                            const ContainerType &container,
                                            const PointType &point) {

  constexpr static bool store_jacobian = PointType::store_jacobian;

  if constexpr (on_device) {
    container.xix(index.ispec, index.iz, index.iy, index.ix) = point.xix;
    container.xiy(index.ispec, index.iz, index.iy, index.ix) = point.xiy;
    container.xiz(index.ispec, index.iz, index.iy, index.ix) = point.xiz;
    container.etax(index.ispec, index.iz, index.iy, index.ix) = point.etax;
    container.etay(index.ispec, index.iz, index.iy, index.ix) = point.etay;
    container.etaz(index.ispec, index.iz, index.iy, index.ix) = point.etaz;
    container.gammax(index.ispec, index.iz, index.iy, index.ix) = point.gammax;
    container.gammay(index.ispec, index.iz, index.iy, index.ix) = point.gammay;
    container.gammaz(index.ispec, index.iz, index.iy, index.ix) = point.gammaz;
    if constexpr (store_jacobian) {
      container.jacobian(index.ispec, index.iz, index.iy, index.ix) =
          point.jacobian;
    }
  } else {
    container.xix(index.ispec, index.iz, index.iy, index.ix) = point.xix;
    container.xiy(index.ispec, index.iz, index.iy, index.ix) = point.xiy;
    container.xiz(index.ispec, index.iz, index.iy, index.ix) = point.xiz;
    container.etax(index.ispec, index.iz, index.iy, index.ix) = point.etax;
    container.etay(index.ispec, index.iz, index.iy, index.ix) = point.etay;
    container.etaz(index.ispec, index.iz, index.iy, index.ix) = point.etaz;
    container.gammax(index.ispec, index.iz, index.iy, index.ix) = point.gammax;
    container.gammay(index.ispec, index.iz, index.iy, index.ix) = point.gammay;
    container.gammaz(index.ispec, index.iz, index.iy, index.ix) = point.gammaz;
    if constexpr (store_jacobian) {
      container.jacobian(index.ispec, index.iz, index.iy, index.ix) =
          point.jacobian;
    }
  }
}

} // namespace specfem::assembly
