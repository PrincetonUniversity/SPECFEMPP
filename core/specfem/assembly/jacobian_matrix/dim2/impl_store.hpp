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

  const auto mask = index.template get_mask<simd>();

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  if constexpr (on_device) {
    Kokkos::Experimental::simd_partial_store(
        jacobian_matrix.xix, &derivatives.xix[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        jacobian_matrix.gammax, &derivatives.gammax[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        jacobian_matrix.xiz, &derivatives.xiz[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        jacobian_matrix.gammaz, &derivatives.gammaz[_index], mask, tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::simd_partial_store(jacobian_matrix.jacobian,
                                               &derivatives.jacobian[_index],
                                               mask, tag_type());
    }
  } else {
    Kokkos::Experimental::simd_partial_store(
        jacobian_matrix.xix, &derivatives.h_xix[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(jacobian_matrix.gammax,
                                             &derivatives.h_gammax[_index],
                                             mask, tag_type());
    Kokkos::Experimental::simd_partial_store(
        jacobian_matrix.xiz, &derivatives.h_xiz[_index], mask, tag_type());
    Kokkos::Experimental::simd_partial_store(jacobian_matrix.gammaz,
                                             &derivatives.h_gammaz[_index],
                                             mask, tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::simd_partial_store(jacobian_matrix.jacobian,
                                               &derivatives.h_jacobian[_index],
                                               mask, tag_type());
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
