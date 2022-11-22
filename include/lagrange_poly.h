#ifndef LAGRANGE_H
#define LAGRANGE_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace Lagrange {
/**
 * @brief Compute lagrange interpolants and its derivatives at xi
 *
 * @param xi Value to calculate lagrange interpolants and its derivatives
 * @param ngll Order used to approximate functions
 * @param xigll GLL points
 * @return std::tuple<specfem::HostView1d<type_real>,
 * specfem::HostView1d<type_real> > values of lagrange interpolants and
 * derivates calculated at xi
 */
std::tuple<specfem::HostView1d<type_real>, specfem::HostView1d<type_real> >
compute_lagrange_interpolants(const type_real xi, const int ngll,
                              const specfem::HostMirror1d<type_real> xigll);

/**
 * @brief Compute the derivatives of Lagrange functions at GLL points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A19)
 *
 * @param xigll GLL points
 * @param ngll Order used to approximate functions
 * @return specfem::HostView2d<type_real> Derivates of lagrange polynomials at
 * GLL points
 */
specfem::HostView2d<type_real>
compute_lagrange_derivatives_GLL(const specfem::HostMirror1d<type_real> xigll,
                                 const int ngll);

/**
 * @brief Compute the derivatives of Jacobi functions at GLJ points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A27)
 *
 * @warning This function hasn't been tested yet. Could be potential source of
 * error.
 *
 * @param xiglj GLJ points
 * @param nglj Order used to approximate functions
 * @return specfem::HostView2d<type_real> Derivates of Jacobi polynomials at GLJ
 * points
 */
specfem::HostView2d<type_real>
compute_jacobi_derivatives_GLJ(const specfem::HostMirror1d<type_real> xiglj,
                               const int nglj);

} // namespace Lagrange

#endif
