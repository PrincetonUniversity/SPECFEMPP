#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

template <typename PointIndexType, typename ContainerType, typename PointType,
          typename std::enable_if_t<
              PointIndexType::dimension_tag == specfem::dimension::type::dim3 &&
                  !(PointIndexType::using_simd == PointType::simd::using_simd),
              int> = 0>
void load_on_host(const PointIndexType &index, const ContainerType &container,
                  PointType &point) {
  static_assert(
      specfem::data_access::CheckCompatibility<PointIndexType, ContainerType,
                                               PointType>::value,
      "Incompatible types");

  constexpr static bool load_jacobian = PointType::store_jacobian;

  point.xix = container.h_xix(index.ispec, index.iz, index.iy, index.ix);
  point.xiy = container.h_xiy(index.ispec, index.iz, index.iy, index.ix);
  point.xiz = container.h_xiz(index.ispec, index.iz, index.iy, index.ix);
  point.etax = container.h_etax(index.ispec, index.iz, index.iy, index.ix);
  point.etay = container.h_etay(index.ispec, index.iz, index.iy, index.ix);
  point.etaz = container.h_etaz(index.ispec, index.iz, index.iy, index.ix);
  point.gammax = container.h_gammax(index.ispec, index.iz, index.iy, index.ix);
  point.gammay = container.h_gammay(index.ispec, index.iz, index.iy, index.ix);
  point.gammaz = container.h_gammaz(index.ispec, index.iz, index.iy, index.ix);
  if constexpr (load_jacobian) {
    point.jacobian =
        container.h_jacobian(index.ispec, index.iz, index.iy, index.ix);
  }
}
