#ifndef GLL_UTILS_H
#define GLL_UTILS_H

#include <tuple>
#include <vector>

namespace gll_utils {
/**
 * @brief Computes the Jacobi polynomial of degree n and its derivative at x
 *
 * @param n degree of the polynomial
 * @param alpha alpha value of Jacobi
 * @param beta beta value
 * @param x value to evaluate the jacobi polynomial and its derivative
 * @return std::tuple<double, double> [p, pd] where p is the value of the
 * polynomial and pd is its derivative at x
 */
std::tuple<double, double> jacobf(const int n, const double alpha,
                                  const double beta, const double x);

/**
 * @brief Compute Gauss points i.e. the zeros of the Jacobi polynomials
 *
 * @param np degree of the Jacobi polynomial
 * @param alpha alpha value of Jacobi polynomial
 * @param beta beta value of Jacobi polynomial
 * @return std::vector<double>& Gauss points as a std::vector (vector.size() ==
 * np)
 */
std::vector<double> &jacg(const int np, const double alpha, const double beta);
} // namespace gll_utils

#endif // GLL_UTILS_H
