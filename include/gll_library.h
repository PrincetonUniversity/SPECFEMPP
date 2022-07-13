#ifndef GLL_LIBRARY_H
#define GLL_LIBRARY_H

#include "gll_utils.h"
#include <array>

namespace gll_library {
/**
 * @brief Compute Legendre polynomial of degree n at point z
 *
 * @param z point to evaluate Legendre polynomial
 * @param n degree of the Legendre polynomial
 * @return double value of Legendre polynomial
 */
double pnleg(const double z, const int n);
/**
 * @brief Compute Gauss-Labatto-Jacobi polynomial of degree n at point z
 *
 * @param z point to evaluate Gauss-Labatto-Jacobi polynomial
 * @param n degree of the Gauss-Labatto-Jacobi polynomial
 * @return double value of Gauss-Labatto-Jacobi polynomial
 */
double pnglj(const double z, const int n);
/**
 * @brief Compute derivative of Legendre polynomial of degree n at point z
 *
 * @param z point to evaluate derivative of Legendre polynomial
 * @param n degree of the Legendre polynomial
 * @return double value of derivative of Legendre polynomial
 */
double pndleg(const double z, const int n);
/**
 * @brief Compute derivative of Gauss-Labatto-Jacobi polynomial of degree n at
 * point z
 *
 * @param z point to evaluate derivative of Gauss-Labatto-Jacobi polynomial
 * @param n degree of the Gauss-Labatto-Jacobi polynomial
 * @return double value of derivative of Gauss-Labatto-Jacobi polynomial
 */
double pndglj(const double z, const int n);
/**
 * @brief Generate np Gauss-Lobatto-Jacobi points and the weights associated
 * with Jacobi polynomials of degree n = np-1.
 * @note alpha and beta coefficients must be greater than -1. Legendre
 * polynomials are special case of Jacobi polynomials just by setting alpha and
 * beta to 0.
 *
 * @param z HostArray where GLL points will be stored
 * @param w HostArray where GLL weights will be stored
 * @param np Number of GLL points
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 */
void zwgljd(Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> z,
            Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> w,
            const int np, const double alpha, const double beta);
} // namespace gll_library

#endif // GLL_LIBRARY_H
