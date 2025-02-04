#pragma once

#include "element/quadrature.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "point/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace benchmarks {

template <typename MemberType, typename ViewType>
KOKKOS_FUNCTION void
load_on_device(const MemberType &team,
               const specfem::compute::quadrature &quadrature,
               ViewType &element_quadrature) {

  constexpr bool store_hprime_gll = ViewType::store_hprime_gll;

  constexpr bool store_weight_times_hprime_gll =
      ViewType::store_weight_times_hprime_gll;
  constexpr int NGLL = ViewType::ngll;

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a host execution space");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);
        if constexpr (store_hprime_gll) {
          element_quadrature.hprime_gll(iz, ix) = quadrature.gll.hprime(iz, ix);
        }
        if constexpr (store_weight_times_hprime_gll) {
          element_quadrature.hprime_wgll(ix, iz) =
              quadrature.gll.hprime(iz, ix) * quadrature.gll.weights(iz);
        }
      });
}

template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              PointPartialDerivativesType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const specfem::point::simd_index<PointPartialDerivativesType::dimension>
        &index,
    const specfem::compute::partial_derivatives &derivatives,
    PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  using simd = typename PointPartialDerivativesType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  Kokkos::Experimental::where(mask, partial_derivatives.xix)
      .copy_from(&derivatives.xix(ispec, iz, ix), tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.gammax)
      .copy_from(&derivatives.gammax(ispec, iz, ix), tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.xiz)
      .copy_from(&derivatives.xiz(ispec, iz, ix), tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.gammaz)
      .copy_from(&derivatives.gammaz(ispec, iz, ix), tag_type());
  if constexpr (StoreJacobian) {
    Kokkos::Experimental::where(mask, partial_derivatives.jacobian)
        .copy_from(&derivatives.jacobian(ispec, iz, ix), tag_type());
  }
}

template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const specfem::point::index<PointPartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  partial_derivatives.xix = derivatives.xix(ispec, iz, ix);
  partial_derivatives.gammax = derivatives.gammax(ispec, iz, ix);
  partial_derivatives.xiz = derivatives.xiz(ispec, iz, ix);
  partial_derivatives.gammaz = derivatives.gammaz(ispec, iz, ix);
  if constexpr (StoreJacobian) {
    partial_derivatives.jacobian = derivatives.jacobian(ispec, iz, ix);
  }
}

} // namespace benchmarks
} // namespace specfem
