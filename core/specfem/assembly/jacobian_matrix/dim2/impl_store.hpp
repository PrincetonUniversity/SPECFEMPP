#pragma once

#include "enumerations/interface_tags.hpp"
#include "specfem/data_access.hpp"
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
inline void impl_store(const IndexType &index, const ContainerType &derivatives,
                       const PointType &jacobian_matrix) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PointType::store_jacobian;

  using simd = typename PointType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  if constexpr (on_device) {
    Kokkos::Experimental::where(mask, jacobian_matrix.xix)
        .copy_to(&derivatives.xix[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.gammax)
        .copy_to(&derivatives.gammax[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.xiz)
        .copy_to(&derivatives.xiz[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.gammaz)
        .copy_to(&derivatives.gammaz[_index], tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, jacobian_matrix.jacobian)
          .copy_to(&derivatives.jacobian[_index], tag_type());
    }
  } else {
    Kokkos::Experimental::where(mask, jacobian_matrix.xix)
        .copy_to(&derivatives.h_xix[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.gammax)
        .copy_to(&derivatives.h_gammax[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.xiz)
        .copy_to(&derivatives.h_xiz[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.gammaz)
        .copy_to(&derivatives.h_gammaz[_index], tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, jacobian_matrix.jacobian)
          .copy_to(&derivatives.h_jacobian[_index], tag_type());
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
inline void impl_store(const IndexType &index, const ContainerType &derivatives,
                       const PointType &jacobian_matrix) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PointType::store_jacobian;

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  if constexpr (on_device) {
    derivatives.xix[_index] = jacobian_matrix.xix;
    derivatives.gammax[_index] = jacobian_matrix.gammax;
    derivatives.xiz[_index] = jacobian_matrix.xiz;
    derivatives.gammaz[_index] = jacobian_matrix.gammaz;
    if constexpr (StoreJacobian) {
      derivatives.jacobian[_index] = jacobian_matrix.jacobian;
    }
  } else {
    derivatives.h_xix[_index] = jacobian_matrix.xix;
    derivatives.h_gammax[_index] = jacobian_matrix.gammax;
    derivatives.h_xiz[_index] = jacobian_matrix.xiz;
    derivatives.h_gammaz[_index] = jacobian_matrix.gammaz;
    if constexpr (StoreJacobian) {
      derivatives.h_jacobian[_index] = jacobian_matrix.jacobian;
    }
  }
}
} // namespace specfem::assembly
