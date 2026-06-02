#pragma once

#include "specfem/datatype.hpp"
#include "specfem/execution.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

/**
 * @file scatter.hpp
 * @brief Scatter algorithms — transpose of the spectral-element divergence
 *
 * Replaces the two-pass gather pattern (store full stress integrand in shared
 * memory, then reduce) with a scatter pattern: each source GLL point atomically
 * accumulates its contribution into a shared acceleration view, cutting shared
 * memory from O(ngll^dim × components × dim) to O(ngll^dim × components).
 */

namespace specfem {
namespace algorithms {
namespace impl {

/**
 * @brief Scatter a stress-integrand tensor at a 2D source GLL point into the
 * shared acceleration scratch — transpose of @c element_divergence for 2D.
 *
 * Source point @c (iz0, ix0) contributes:
 *   - x-direction: @c acc[iz0][ix] += w(iz0)*F[c][0]*H'_ξ(ix0,ix)*w(ix0)
 *     for all @c ix in [0, ngll)
 *   - z-direction: @c acc[iz][ix0] += w(ix0)*F[c][1]*H'_γ(iz0,iz)*w(iz0)
 *     for all @c iz in [0, ngll)
 *
 * @tparam TensorPointViewType  TensorPointViewType<type_real,components,2,simd>
 * @tparam WeightsType          Quadrature weights accessor (operator()(int))
 * @tparam QuadratureType       Lagrange derivative accessor (.xi(i,j),
 * .gamma(i,j))
 * @tparam ChunkAccType         Chunk acceleration scratch type
 * @param F            Stress integrand @c F = stress × jacobian at source point
 * @param local_index  2D source GLL index
 * @param weights      Quadrature weights
 * @param hprime       Lagrange derivative polynomials
 * @param acc          Shared scratch acceleration view (non-const, atomic
 * writes)
 */
template <typename TensorPointViewType, typename T, typename QuadratureType,
          typename ChunkAccType>
KOKKOS_FORCEINLINE_FUNCTION void scattered_divergence(
    const TensorPointViewType &F,
    const specfem::point::index<specfem::element::dimension_tag::dim2,
                                TensorPointViewType::using_simd> &local_index,
    const T &w_product, const QuadratureType &hprime, ChunkAccType &acc) {

  constexpr int ngll = ChunkAccType::ngll;
  constexpr int components = TensorPointViewType::components;
  const int ielement = local_index.ispec;
  const int iz0 = local_index.iz;
  const int ix0 = local_index.ix;

  // x-direction: F[c][0] at (iz0,ix0) scatters to acc[iz0][ix] for all ix
  for (int ix = 0; ix < ngll; ++ix)
    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::atomic_add(&acc(ielement, iz0, ix, icomp),
                         F(icomp, 0) * hprime.xi(ix0, ix) * w_product);

  // z-direction: F[c][1] at (iz0,ix0) scatters to acc[iz][ix0] for all iz
  for (int iz = 0; iz < ngll; ++iz)
    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::atomic_add(&acc(ielement, iz, ix0, icomp),
                         w_product * F(icomp, 1) * hprime.gamma(iz0, iz));
}

/**
 * @brief Scatter a stress-integrand tensor at a 3D source GLL point into the
 * shared acceleration scratch — transpose of @c element_divergence for 3D.
 *
 * Source point @c (iz0, iy0, ix0) contributes:
 *   - x-direction: @c acc[iz0][iy0][ix] +=
 * w(iz0)*w(iy0)*F[c][0]*H'_ξ(ix0,ix)*w(ix0)
 *   - y-direction: @c acc[iz0][iy][ix0] +=
 * w(iz0)*w(ix0)*F[c][1]*H'_η(iy0,iy)*w(iy0)
 *   - z-direction: @c acc[iz][iy0][ix0] +=
 * w(iy0)*w(ix0)*F[c][2]*H'_γ(iz0,iz)*w(iz0)
 *
 * @tparam TensorPointViewType  TensorPointViewType<type_real,components,3,simd>
 * @tparam WeightsType          Quadrature weights accessor
 * @tparam QuadratureType       Lagrange derivative accessor (.xi, .eta, .gamma)
 * @tparam ChunkAccType         Chunk acceleration scratch type
 */
template <typename TensorPointViewType, typename T, typename QuadratureType,
          typename ChunkAccType>
KOKKOS_FORCEINLINE_FUNCTION void scattered_divergence(
    const TensorPointViewType &F,
    const specfem::point::index<specfem::element::dimension_tag::dim3,
                                TensorPointViewType::using_simd> &local_index,
    const T &w_product, const QuadratureType &hprime, ChunkAccType &acc) {

  constexpr int ngll = ChunkAccType::ngll;
  constexpr int components = TensorPointViewType::components;
  const int ielement = local_index.ispec;
  const int iz0 = local_index.iz;
  const int iy0 = local_index.iy;
  const int ix0 = local_index.ix;

  // x-direction: scatters to acc[iz0][iy0][ix] for all ix
  for (int ix = 0; ix < ngll; ++ix)
    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::atomic_add(&acc(ielement, iz0, iy0, ix, icomp),
                         w_product * F(icomp, 0) * hprime.xi(ix0, ix));

  // y-direction: scatters to acc[iz0][iy][ix0] for all iy
  for (int iy = 0; iy < ngll; ++iy)
    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::atomic_add(&acc(ielement, iz0, iy, ix0, icomp),
                         w_product * F(icomp, 1) * hprime.eta(iy0, iy));

  // z-direction: scatters to acc[iz][iy0][ix0] for all iz
  for (int iz = 0; iz < ngll; ++iz)
    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::atomic_add(&acc(ielement, iz, iy0, ix0, icomp),
                         w_product * F(icomp, 2) * hprime.gamma(iz0, iz));
}

} // namespace impl
} // namespace algorithms
} // namespace specfem
