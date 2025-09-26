#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

template <
    typename PointIndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<PointIndexType>::value &&
            PointIndexType::dimension_tag == specfem::dimension::type::dim3 &&
            !PointIndexType::using_simd && !PointType::simd::using_simd &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const PointIndexType &index,
                                                const ContainerType &container,
                                                PointType &point) {
  static_assert(
      specfem::data_access::CheckCompatibility<PointIndexType, ContainerType,
                                               PointType>::value,
      "Incompatible types");

  constexpr static bool load_jacobian = PointType::store_jacobian;

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
}

} // namespace specfem::assembly
