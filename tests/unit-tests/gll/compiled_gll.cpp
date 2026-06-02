#include "specfem/quadrature/compiletime/lagrange.hpp"
#include "specfem/quadrature/compiletime/legendre.hpp"
#include "specfem/quadrature/compiletime/rational_polynomial.hpp"

#include "specfem/quadrature/gll.hpp"
#include "specfem/utilities/is_close.hpp"

#include "test_macros.hpp"

#include <array>
#include <gtest/gtest.h>
#include <type_traits>

#include <Kokkos_Core.hpp>

// ===============================================
// Legendre polynomial evaluations
// ===============================================
static_assert(
    std::is_same_v<
        specfem::quadrature::compiletime::LegendrePolynomial<2>::coefficients,
        std::tuple<std::ratio<-1, 2>, std::ratio<0>, std::ratio<3, 2>>>,
    "LegendrePolynomial<2> should be 3/2 x^2 - 1");

static_assert(
    std::is_same_v<
        specfem::quadrature::compiletime::LegendrePolynomial<3>::coefficients,
        std::tuple<std::ratio<0>, std::ratio<-3, 2>, std::ratio<0>,
                   std::ratio<5, 2>>>,
    "LegendrePolynomial<3> should be 5/2 x^3 - 3 x");

// ===============================================
// Rational Polynomial differentiation
// ===============================================

static_assert(
    std::is_same_v<specfem::quadrature::compiletime::RationalPolynomial<
                       std::ratio<1>, std::ratio<2>, std::ratio<2>,
                       std::ratio<3>>::derivative,
                   specfem::quadrature::compiletime::RationalPolynomial<
                       std::ratio<2>, std::ratio<4>, std::ratio<9>>>,
    "derivative of 3x^3 + 2x^2 + 2x + 1 should be 9x^2 + 4x + 2");

template <typename T, size_t N, template <typename, size_t> typename Arr1>
constexpr bool array_close(const Arr1<T, N> &a, const std::array<T, N> &b) {
  for (int i = 0; i < N; i++) {
    if (specfem::quadrature::compiletime::impl::fabs(a[i] - b[i]) > 1e-10) {
      return false;
    }
  }
  return true;
}

// ===============================================
// Rootfinding: Legendre<7>
// ===============================================
static_assert(
    array_close(
        decltype(specfem::quadrature::compiletime::RationalPolynomialWithRoots(
            specfem::quadrature::compiletime::LegendrePolynomial<
                7>::coefficients()))::roots,
        { -0.9491079123427583752459213428664952516556,
          -0.7415311855993944600839995473506860435009,
          -0.4058451513773971841558818596240598708391, 0.0,
          0.4058451513773971841558818596240598708391,
          0.7415311855993944600839995473506860435009,
          0.9491079123427583752459213428664952516556 }),
    "Legendre<7> roots as given.");

// ===============================================
// GLL nodes correct?
// ===============================================
static_assert(array_close(specfem::quadrature::compiletime::gll_initializer<
                              7, double>::get_nodes(),
                          { -1, -0.8302238962785670750577082799281924962997,
                            -0.4688487934707141757684212279855273663998, 0.0,
                            0.4688487934707141757684212279855273663998,
                            0.8302238962785670750577082799281924962997, 1 }),
              "Did not recover GLL6 (NGLL=7) nodes correctly");

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
