#include <Kokkos_Core.hpp>

namespace Lagrange {

void compute_lagrange_interpolants(
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> h,
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> hprime,
    const double xi, const int ngll,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xigll);
void compute_lagrange_derivatives_GLL(
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> hprime_ii,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xigll,
    const int ngll);

} // namespace Lagrange
