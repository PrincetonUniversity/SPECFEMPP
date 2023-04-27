#ifndef LAGRANGE_H
#define LAGRANGE_H

#include "../include/kokkos_abstractions.h"
#include "../include/specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace quadrature {
namespace gll {
namespace Lagrange {
/**
 * @brief Compute lagrange interpolants and its derivatives at xi
 *
 * @param xi Value to calculate lagrange interpolants and its derivatives
 * @param ngll Order used to approximate functions
 * @param xigll GLL points
 * @return std::tuple<specfem::kokkos::HostView1d<type_real>,
 * specfem::kokkos::HostView1d<type_real> > values of lagrange interpolants and
 * derivates calculated at xi
 */
std::tuple<specfem::kokkos::HostView1d<type_real>,
           specfem::kokkos::HostView1d<type_real> >
compute_lagrange_interpolants(
    const type_real xi, const int ngll,
    const specfem::kokkos::HostView1d<type_real> xigll);

/**
 * @brief Compute the derivatives of Lagrange functions at GLL points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A19)
 *
 * @param xigll GLL points
 * @param ngll Order used to approximate functions
 * @return specfem::kokkos::HostView2d<type_real> Derivates of lagrange
 * polynomials at GLL points
 */
specfem::kokkos::HostView2d<type_real> compute_lagrange_derivatives_GLL(
    const specfem::kokkos::HostView1d<type_real> xigll, const int ngll);

/**
 * @brief Compute the derivatives of Jacobi functions at GLJ points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A27)
 *
 * @warning This function hasn't been tested yet. Could be potential source of
 * error.
 *
 * @param xiglj GLJ points
 * @param nglj Order used to approximate functions
 * @return specfem::kokkos::HostView2d<type_real> Derivates of Jacobi
 * polynomials at GLJ points
 */
specfem::kokkos::HostView2d<type_real> compute_jacobi_derivatives_GLJ(
    const specfem::kokkos::HostView1d<type_real> xiglj, const int nglj);
/**
 * @brief Compute lagrange interpolants and its derivatives at xi
 *
 * @param h Values of lagrange interpolants calculated at xi
 * @param hprime Values of derivatives of lagrange interpolants calculated at xi
 * @param xi Value to calculate lagrange interpolants and its derivatives
 * @param ngll Order used to approximate functions
 * @param xigll GLL points
 */
void compute_lagrange_interpolants(
    specfem::kokkos::HostMirror1d<type_real> h,
    specfem::kokkos::HostMirror1d<type_real> hprime, const type_real xi,
    const int ngll, const specfem::kokkos::HostMirror1d<type_real> xigll);

/**
 * @brief Compute the derivatives of Lagrange functions at GLL points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A19)
 *
 * @param hprime_ii Derivates of lagrange polynomials at GLL points
 * @param xigll GLL points
 * @param ngll Order used to approximate functions
 */
void compute_lagrange_derivatives_GLL(
    specfem::kokkos::HostMirror2d<type_real> hprime_ii,
    const specfem::kokkos::HostMirror1d<type_real> xigll, const int ngll);

/**
 * @brief Compute the derivatives of Jacobi functions at GLJ points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A27)
 *
 * @warning This function hasn't been tested yet. Could be potential source of
 * error.
 *
 * @param hprimeBar_ii Derivates of Jacobi polynomials at GLJ points
 * @param xiglj GLJ points
 * @param nglj Order used to approximate functions
 */
void compute_jacobi_derivatives_GLJ(
    specfem::kokkos::HostMirror2d<type_real> hprimeBar_ii,
    const specfem::kokkos::HostMirror1d<type_real> xiglj, const int nglj);

} // namespace Lagrange
} // namespace gll
} // namespace quadrature
} // namespace specfem

#endif
