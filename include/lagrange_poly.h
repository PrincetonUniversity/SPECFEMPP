#ifndef LAGRANGE_H
#define LAGRANGE_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

using HostMirror1d = specfem::HostMirror1d<type_real>;
using HostMirror2d = specfem::HostMirror2d<type_real>;

namespace Lagrange {
/**
 * @brief Compute lagrange interpolants and its derivatives at xi
 *
 * @param h HostMirror1d Values of lagrange interpolants calculated at xi i.e.
 * h[i] = l_{i}(xi) && h.extent() == N && h.rank() == 1
 * @param hprime HostMirror1d Values of derivatives of lagrange interpolants
 * calculated at xi i.e. h[i] = l_{i}(xi) && h.extent() == N && h.rank() == 1
 * @param xi Value to calculate lagrange interpolants and its derivatives
 * @param ngll Order used to approximate functions
 * @param xigll HostMirror1d Array of GLL points generally calculated using
 * gll_library::zwgljd
 */
void compute_lagrange_interpolants(HostMirror1d h, HostMirror1d hprime,
                                   const type_real xi, const int ngll,
                                   const HostMirror1d xigll);

/**
 * @brief Compute the derivatives of Lagrange functions at GLL points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A19)
 *
 * @param hprime_ii HostMirror2d Derivates of lagrange polynomials at GLL points
 * i.e. \f$h'(i,j) = \partial_{\xi}l_{j}(\xi_i)\f$ where \f$\xi_i \epsilon
 * \textrm{GLL points}\f$
 * @param xigll HostMirror1d GLL points generally calculated using
 * gll_library::zwgljd
 * @param ngll Order used to approximate functions
 */
void compute_lagrange_derivatives_GLL(HostMirror2d hprime_ii,
                                      const HostMirror1d xigll, const int ngll);

/**
 * @brief Compute the derivatives of Jacobi functions at GLJ points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A27)
 *
 * @warning This function hasn't been tested yet. Could be potential source of
 * error.
 *
 * @param hprimeBar_ii HostMirror2d Derivates of Jacobi polynomials at GLJ
 * points at \f$
 * (\alpha, \beta) = (0,1) i.e. \f$h'(i,j) = \partial_{\xi}l_{j}(\xi_i)\f$ where
 * \f$\xi_i \epsilon \textrm{GLJpoints}\f$
 * @param xiglj HostMirror1d GLJ points generally calculated using
 * gll_library::zwgljd
 * @param nglj Order used to approximate functions
 */
void compute_jacobi_derivatives_GLJ(HostMirror2d hprimeBar_ii,
                                    const HostMirror1d xiglj, const int nglj);

} // namespace Lagrange

#endif
