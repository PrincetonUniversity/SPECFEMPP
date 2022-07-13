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
void zwgljd(HostArray<double> z, HostArray<double> w, const int np,
            const double alpha, const double beta);

/**
 * @warning GLL class is still in progress,
 * will get to it as I understand the Fortran code more
 * and what things need to be added here
 *
 */
class gll {
public:
  gll();
  gll(const double alpha, const double beta);
  gll(const double alpha, const double beta, const int ngll);
  gll(const double alpha, const double beta, const int ngllx, const int ngllz);

private:
  double alpha, beta;
  int ngllx, ngllz;
};
} // namespace gll_library

#endif // GLL_LIBRARY_H
