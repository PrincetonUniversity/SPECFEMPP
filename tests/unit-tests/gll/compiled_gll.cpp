#include "specfem/quadrature/compiletime/lagrange.hpp"

#include "specfem/quadrature/gll.hpp"
#include "specfem/utilities/is_close.hpp"

#include "test_macros.hpp"

#include <array>
#include <gtest/gtest.h>
#include <type_traits>

#include <Kokkos_Core.hpp>

/**
 * @brief Tests kokkos kernel access for nodes and lagrange polynomial
 * evaluations.
 */
template <int NGLL> void verify_against_runtime_on_device() {
  specfem::quadrature::gll::gll GLL(0, 0, NGLL);
  const auto runtime_xi = GLL.get_xi();
  const auto runtime_hxi = GLL.get_hxi();

  Kokkos::View<type_real[NGLL]> xi_vals("xi_vals");
  Kokkos::View<type_real[NGLL][NGLL]> lagrange_evaluations(
      "lagrange_evaluations");

  using CompiledGLL = specfem::quadrature::compiletime::gll<NGLL>;
  const typename CompiledGLL::Nodes compiled_nodes;

  Kokkos::parallel_for(
      "impl_compute_coupling_test", NGLL, KOKKOS_LAMBDA(const int &iworker) {
        xi_vals(iworker) = compiled_nodes(iworker);

        Kokkos::Array<type_real, NGLL> L;
        CompiledGLL::eval_all(runtime_xi(iworker), L);
        for (int i = 0; i < NGLL; i++) {
          lagrange_evaluations(i, iworker) = L[i];
        }
      });

  // verify knots are equal
  const auto h_xi_vals = Kokkos::create_mirror(xi_vals);
  Kokkos::deep_copy(h_xi_vals, xi_vals);
  for (int igll = 0; igll < NGLL; igll++) {
    EXPECT_TRUE(
        specfem::utilities::is_close(h_xi_vals(igll), runtime_hxi(igll)))
        << "node values (NGLL = " << NGLL << "): quadrature point " << igll
        << "\n"
        << expected_got(runtime_hxi(igll), h_xi_vals(igll));
  }

  // verify kronecker delta of L
  const auto h_lagrange_evaluations =
      Kokkos::create_mirror(lagrange_evaluations);
  Kokkos::deep_copy(h_lagrange_evaluations, lagrange_evaluations);
  for (int ipoly = 0; ipoly < NGLL; ipoly++) {
    for (int igll = 0; igll < NGLL; igll++) {
      type_real expected = ipoly == igll ? 1 : 0;
      EXPECT_TRUE(specfem::utilities::is_close(
          h_lagrange_evaluations(ipoly, igll), expected, (type_real)1e-5,
          (type_real)1e-4))
          << "Lagrange polynomial evaluations at nodes (NGLL = " << NGLL
          << "): Lagrange polynomial " << ipoly << ", quadrature point " << igll
          << "\n"
          << expected_got(expected, h_lagrange_evaluations(ipoly, igll));
    }
  }
}

TEST(CompiledGLL, EquateToRuntimeDeviceViews) {
  verify_against_runtime_on_device<5>();
  verify_against_runtime_on_device<8>();
  verify_against_runtime_on_device<10>();
}
