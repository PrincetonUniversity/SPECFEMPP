#pragma once

#include "gll_utils.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <array>

namespace specfem {
namespace quadrature {
namespace gll {
namespace gll_library {
/**
 * Compute Legendre polynomial of degree n at point z
 *
 * @param z point to evaluate Legendre polynomial
 * @param n degree of the Legendre polynomial
 * @return type_real value of Legendre polynomial
 */
type_real pnleg(const type_real z, const int n);
/**
 * Compute Gauss-Labatto-Jacobi polynomial of degree n at point z
 *
 * @param z point to evaluate Gauss-Labatto-Jacobi polynomial
 * @param n degree of the Gauss-Labatto-Jacobi polynomial
 * @return type_real value of Gauss-Labatto-Jacobi polynomial
 */
type_real pnglj(const type_real z, const int n);
/**
 * Compute derivative of Legendre polynomial of degree n at point z
 *
 * @param z point to evaluate derivative of Legendre polynomial
 * @param n degree of the Legendre polynomial
 * @return type_real value of derivative of Legendre polynomial
 */
type_real pndleg(const type_real z, const int n);
/**
 * Compute derivative of Gauss-Labatto-Jacobi polynomial of degree n at
 * point z
 *
 * @param z point to evaluate derivative of Gauss-Labatto-Jacobi polynomial
 * @param n degree of the Gauss-Labatto-Jacobi polynomial
 * @return type_real value of derivative of Gauss-Labatto-Jacobi polynomial
 */
type_real pndglj(const type_real z, const int n);
/**
 * Generate np Gauss-Lobatto-Jacobi points and the weights associated
 * with Jacobi polynomials of degree n = np-1.
 * @note alpha and beta coefficients must be greater than -1. Legendre
 * polynomials are special case of Jacobi polynomials just by setting alpha and
 * beta to 0.
 *
 * @param z HostMirror1d where GLL points will be stored
 * @param w HostMirror1d where GLL weights will be stored
 * @param np Number of GLL points
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 */
void zwgljd(Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> z,
            Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> w,
            const int np, const type_real alpha, const type_real beta);
/**
 * Generate np Gauss-Lobatto-Jacobi points and the weights associated
 * with Jacobi polynomials of degree n = np-1.
 * @note alpha and beta coefficients must be greater than -1. Legendre
 * polynomials are special case of Jacobi polynomials just by setting alpha and
 * beta to 0.
 *
 * @param np Number of GLL points
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 * @return std::tuple<specfem::HostView<type_real>,
 * specfem::HostView<type_real>> GLL points and weights
 */
std::tuple<Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>,
           Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> >
zwgljd(const int np, const type_real alpha, const type_real beta);
} // namespace gll_library
} // namespace gll
} // namespace quadrature
} // namespace specfem
