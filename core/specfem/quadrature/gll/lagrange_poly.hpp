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
 * @return std::tuple<Kokkos::View<type_real *, Kokkos::LayoutRight,
 * Kokkos::HostSpace>, Kokkos::View<type_real *, Kokkos::LayoutRight,
 * Kokkos::HostSpace> > values of lagrange interpolants and its derivates
 * calculated at xi
 */
std::tuple<Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>,
           Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> >
compute_lagrange_interpolants(
    const type_real xi, const int ngll,
    const Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
        xigll);

/**
 * @brief Compute the derivatives of Lagrange functions at GLL points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A19)
 *
 * @param xigll GLL points
 * @param ngll Order used to approximate functions
 * @return Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
 * Derivates of lagrange polynomials at GLL points
 */
Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
compute_lagrange_derivatives_GLL(
    const Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
        xigll,
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
 * @return Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
 * Derivates of Jacobi polynomials at GLJ points
 */
Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
compute_jacobi_derivatives_GLJ(
    const Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
        xiglj,
    const int nglj);
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
    Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> h,
    Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace> hprime,
    const type_real xi, const int ngll,
    const Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
        xigll);

/**
 * @brief Compute the derivatives of Lagrange functions at GLL points
 * @note Please refer Nisser-Meyer et.al. 2007 equation (A19)
 *
 * @param hprime_ii Derivates of lagrange polynomials at GLL points
 * @param xigll GLL points
 * @param ngll Order used to approximate functions
 */
void compute_lagrange_derivatives_GLL(
    Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
        hprime_ii,
    const Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
        xigll,
    const int ngll);

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
    Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
        hprimeBar_ii,
    const Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
        xiglj,
    const int nglj);

} // namespace Lagrange
} // namespace gll
} // namespace quadrature
} // namespace specfem

#endif
