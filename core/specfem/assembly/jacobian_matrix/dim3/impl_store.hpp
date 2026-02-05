#pragma once

#include "enumerations/interface_tags.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

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
