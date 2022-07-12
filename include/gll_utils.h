#ifndef GLL_UTILS_H
#define GLL_UTILS_H

#include <Kokkos_Core.hpp>
#include <tuple>

template <typename T>
using HostArray = Kokkos::View<T *, Kokkos::LayoutRight, Kokkos::HostSpace>;
/**
 * @note These routines are primarily called within GLL library module.
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
 * @return std::tuple<double, double> [p, pd] where p is the value of the
 * polynomial and pd is its derivative at x
 */
std::tuple<double, double, double> jacobf(const int n, const double alpha,
                                          const double beta, const double x);

/**
 * @brief Compute Gauss points i.e. the zeros of the Jacobi polynomials
 *
 * @param np degree of the Jacobi polynomial
 * @param alpha alpha value of Jacobi polynomial
 * @param beta beta value of Jacobi polynomial
 * @param xjac HostArray where Gauss points (GLL points) will be stored
 * xjac.extent(0) == np, xjac.rank == 1
 */
void jacg(HostArray<double> &xjac, const int np, const double alpha,
          const double beta);

double calc_gammaf(const double x);
double calc_pnormj(const int n, const double alpha, const double beta);

/**
 * @brief Compute the weights of GLL quadrature at GLL points
 *
 * @param z HostArray of GLL points (use gll_utils::jacg to calculate GLL
 * points)
 * @param w HostArray where the GLL weights will be stored
 * @param np degree of the Jacobi polynomial
 * @param alpha alpha value of Jacobi polynomial
 * @param beta beta value of Jacobi polynomial
 */
void jacw(HostArray<double> z, HostArray<double> w, const int np,
          const int alpha, const int beta);

/**
 * @brief Compute the GLL points and weights of jacobi polynomials inside the
 * open interval (-1,1)
 * @note The weights are not normalized by the this function w(i) =
 * w(i)*(1-z(i)**2)
 * @note This function uses gll_utils::jacg and gll_utils:jacw for calculate the
 * GLL points and weights
 * @param z HostArray where GLL points will be stored
 * @param w HostArray where GLL weights will be stored
 * @param np Degree of the Jacobi polynomial
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 */
void zwgjd(HostArray<double> z, HostArray<double> w, const int np,
           const int alpha, const int beta);

/**
 * @brief Calculate the weight contribution at xi == -1.0
 *
 * @param n NGLL - 1
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 * @return double weight at xi == -1
 */
double endw1(const int n, const double alpha, const double beta);

/**
 * @brief Calculate the weight contribution at xi == 1.0
 *
 * @param n NGLL - 1
 * @param alpha Alpha value of the Jacobi polynomial
 * @param beta Beta value of the Jacobi polynomial
 * @return double weight at xi == 1
 */
double endw2(const int n, const double alpha, const double beta);
} // namespace gll_utils

#endif // GLL_UTILS_H
