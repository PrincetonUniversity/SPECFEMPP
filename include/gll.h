#include <Kokkos_Core.hpp>

namespace gll {
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
  void set_derivation_matrices();
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_xigll() const;
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_zigll() const;
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_wxgll() const;
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_wzgll() const;
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_hprime_xx() const;
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_hprime_zz() const;

private:
  double alpha, beta;
  int ngllx, ngllz;
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xigll, zigll,
      wxgll, wzgll;
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> hprime_xx,
      hprime_zz;
  void set_allocations();
};
} // namespace gll