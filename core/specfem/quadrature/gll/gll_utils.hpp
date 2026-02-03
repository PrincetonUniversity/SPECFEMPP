#pragma once

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>

namespace specfem {
namespace quadrature {
namespace gll {
/**
 * @warning These routines are primarily called within GLL library module.
 * If you require any of the routines here then check if your task can be
 * achieved using GLL Library module
 *
 */
namespace gll_utils {
/**
 * @brief Computes the Jacobi polynomial of degree n and its derivative at x
 *
 * @param n degree of the polynomial
 * @param alpha alpha value of Jacobi
 * @param beta beta value
 * @param x value to evaluate the jacobi polynomial and its derivative
 * @return std::tuple<type_real, type_real> [p, pd] where p is the value of the
 * polynomial and pd is its derivative at x
 */
std::tuple<type_real, type_real, type_real> jacobf(const int n,
                                                   const type_real alpha,
                                                   const type_real beta,
                                                   const type_real x);

/**
 * @brief Compute Gauss points i.e. the zeros of the Jacobi polynomials
 *
 * @param np degree of the Jacobi polynomial
 * @param alpha alpha value of Jacobi polynomial
 * @param beta beta value of Jacobi polynomial
 * @param xjac HostMirror1d where Gauss points (GLL points) will be stored
 * xjac.extent(0) == np, xjac.rank == 1
 */
void jacg(
    Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> xjac,
    const int np, const type_real alpha, const type_real beta);

type_real calc_gammaf(const type_real x);
type_real calc_pnormj(const int n, const type_real alpha, const type_real beta);

/**
 * @brief Compute the weights of GLL quadrature at GLL points
 *
 * @param z HostMirror1d of GLL points (use gll_utils::jacg to calculate GLL
 * points)
 * @param w HostMirror1d where the GLL weights will be stored
 * @param np degree of the Jacobi polynomial
 * @param alpha alpha value of Jacobi polynomial
 * @param beta beta value of Jacobi polynomial
 */
void jacw(Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> z,
          Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> w,
          const int np, const int alpha, const int beta);

/**
 * @brief Compute the GLL points and weights of jacobi polynomials inside the
 * open interval (-1,1)
 * @warning The weights are not normalized by the this function w(i) =
 * w(i)*(1-z(i)**2)
 * @note This function uses gll_utils::jacg and gll_utils::jacw for calculate
 * the GLL points and weights
 * @param z HostArray where GLL points will be stored
 * @param w HostArray where GLL weights will be stored
 * @param np Degree of the Jacobi polynomial
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 */
void zwgjd(Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> z,
           Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> w,
           const int np, const int alpha, const int beta);

/**
 * @brief Calculate the weight contribution at xi == -1.0
 *
 * @param n NGLL - 1
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 * @return type_real weight at xi == -1
 */
type_real endw1(const int n, const type_real alpha, const type_real beta);

/**
 * @brief Calculate the weight contribution at xi == 1.0
 *
 * @param n NGLL - 1
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 * @return type_real weight at xi == 1
 */
type_real endw2(const int n, const type_real alpha, const type_real beta);
} // namespace gll_utils
} // namespace gll
} // namespace quadrature
} // namespace specfem
