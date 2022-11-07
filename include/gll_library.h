#ifndef GLL_LIBRARY_H
#define GLL_LIBRARY_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "gll_utils.h"
#include <array>

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
 * @param z HostArray where GLL points will be stored
 * @param w HostArray where GLL weights will be stored
 * @param np Number of GLL points
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 */
void zwgljd(specfem::HostMirror1d<type_real> z,
            specfem::HostMirror1d<type_real> w, const int np,
            const type_real alpha, const type_real beta);
} // namespace gll_library

#endif // GLL_LIBRARY_H
