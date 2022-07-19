#include <Kokkos_Core.hpp>

namespace Lagrange {
/**
 * @brief Compute lagrange interpolants and its derivatives at xi
 *
 * @param h Values of lagrange interpolants calculated at xi i.e. h[i] =
 * l_{i}(xi) && h.extent() == N && h.rank() == 1
 * @param hprime Values of derivatives of lagrange interpolants calculated at xi
 * i.e. h[i] = l_{i}(xi) && h.extent() == N && h.rank() == 1
 * @param xi Value to calculate lagrange interpolants and its derivatives
 * @param ngll Order used to approximate functions
 * @param xigll Array of GLL points generally calculated using
 * gll_library::zwgljd
 */
void compute_lagrange_interpolants(
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> h,
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> hprime,
    const double xi, const int ngll,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xigll);

/**
 * @brief Compute the derivatives of Lagrange functions at GLL points
 * @note Please refer Nisser-Meyer et.al. 2007 \cite
 * Nisser-meyer-founier-dahlen-2007 equation (A19)
 *
 * @param hprime_ii Derivates of lagrange polynomials at GLL points i.e.
 * hprime_ii(i,j) = \partial_{\xi}l_{j}(xigll(i))
 * @param xigll GLL points generally calculated using gll_library::zwgljd
 * @param ngll Order used to approximate functions
 */
void compute_lagrange_derivatives_GLL(
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> hprime_ii,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xigll,
    const int ngll);

/**
 * @brief Compute the derivatives of Jacobi functions at GLJ points
 * @note Please refer Nisser-Meyer et.al. 2007 \cite
 * Nisser-meyer-founier-dahlen-2007 equation (A27)
 *
 * @param hprimeBar_ii Derivates of Jacobi polynomials at GLJ points i.e.
 * hprime_ii(i,j) = \partial_{\xi}l_{j}(xiglj(i))
 * @param xiglj GLJ points generally calculated using gll_library::zwgljd
 * @param nglj Order used to approximate functions
 */
void compute_jacobi_derivatives_GLJ(
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>
        hprimeBar_ii,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xiglj,
    const int nglj);

} // namespace Lagrange
