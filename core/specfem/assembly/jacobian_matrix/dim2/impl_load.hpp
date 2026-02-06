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

  const auto &mapping = container.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  if constexpr (on_device) {
    Kokkos::Experimental::where(mask, point.xix)
        .copy_from(&container.xix[_index], tag_type());
    Kokkos::Experimental::where(mask, point.gammax)
        .copy_from(&container.gammax[_index], tag_type());
    Kokkos::Experimental::where(mask, point.xiz)
        .copy_from(&container.xiz[_index], tag_type());
    Kokkos::Experimental::where(mask, point.gammaz)
        .copy_from(&container.gammaz[_index], tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, point.jacobian)
          .copy_from(&container.jacobian[_index], tag_type());
    }
  } else {
    Kokkos::Experimental::where(mask, point.xix)
        .copy_from(&container.h_xix[_index], tag_type());
    Kokkos::Experimental::where(mask, point.gammax)
        .copy_from(&container.h_gammax[_index], tag_type());
    Kokkos::Experimental::where(mask, point.xiz)
        .copy_from(&container.h_xiz[_index], tag_type());
    Kokkos::Experimental::where(mask, point.gammaz)
        .copy_from(&container.h_gammaz[_index], tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, point.jacobian)
          .copy_from(&container.h_jacobian[_index], tag_type());
    }
  }
}

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

  const auto &mapping = container.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  if constexpr (on_device) {
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        xix = container.xix.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        gammax = container.gammax.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        xiz = container.xiz.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        gammaz = container.gammaz.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        jacobian = container.jacobian.get_base_view();

    point.xix = xix(_index);
    point.gammax = gammax(_index);
    point.xiz = xiz(_index);
    point.gammaz = gammaz(_index);
    if constexpr (StoreJacobian) {
      point.jacobian = jacobian(_index);
    }
  } else {
    point.xix = container.h_xix[_index];
    point.gammax = container.h_gammax[_index];
    point.xiz = container.h_xiz[_index];
    point.gammaz = container.h_gammaz[_index];
    if constexpr (StoreJacobian) {
      point.jacobian = container.h_jacobian[_index];
    }
  }
}

} // namespace specfem::assembly
