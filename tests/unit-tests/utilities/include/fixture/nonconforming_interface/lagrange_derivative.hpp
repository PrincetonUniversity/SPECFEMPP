#pragma once

#include "../impl/descriptions.hpp"
#include "initializers.hpp"
#include "specfem_setup.hpp"

namespace specfem::test_fixture {

template <typename QuadraturePointsX, typename QuadraturePointsZ>
struct LagrangeDerivative2D {

  static constexpr int ngllx = QuadraturePointsX::nquad;
  static constexpr int ngllz = QuadraturePointsZ::nquad;

  using QuadratureRuleX = QuadratureRule<QuadraturePointsX>;
  using QuadratureRuleZ = QuadratureRule<QuadraturePointsZ>;

private:
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

public:
  Kokkos::View<type_real[ngllx][ngllx], memory_space> xi;
  Kokkos::View<type_real[ngllz][ngllz], memory_space> gamma;

public:
  KOKKOS_FUNCTION LagrangeDerivative2D() = default;
  LagrangeDerivative2D(const std::string &name)
      : xi(name + "(xi)"), gamma(name + "(gamma)") {

    auto h_xi = Kokkos::create_mirror_view(xi);
    auto h_ga = Kokkos::create_mirror_view(gamma);

    for (int ipoly = 0; ipoly < ngllx; ipoly++) {
      for (int isample = 0; isample < ngllx; isample++) {
        h_xi(isample, ipoly) = QuadratureRuleX::evaluate_lagrange_derivative(
            ipoly, QuadraturePointsX::quadrature_points[isample]);
      }
    }

    for (int ipoly = 0; ipoly < ngllz; ipoly++) {
      for (int isample = 0; isample < ngllz; isample++) {
        h_ga(isample, ipoly) = QuadratureRuleZ::evaluate_lagrange_derivative(
            ipoly, QuadraturePointsZ::quadrature_points[isample]);
      }
    }

    Kokkos::deep_copy(xi, h_xi);
    Kokkos::deep_copy(gamma, h_ga);
  }
};
} // namespace specfem::test_fixture
