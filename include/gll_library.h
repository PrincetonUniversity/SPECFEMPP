#ifndef GLL_LIBRARY_H
#define GLL_LIBRARY_H

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
 * @brief
 *
 */
class gll {
public:
  gll();
  gll(const double alpha, const double beta, const int ngll);

private:
  double alpha, beta;
  int ngll;
};
} // namespace gll_library

#endif // GLL_LIBRARY_H
