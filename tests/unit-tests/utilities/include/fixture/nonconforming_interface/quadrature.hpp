#pragma once

#include "../impl/descriptions.hpp"
#include "initializers.hpp"
#include "specfem/setup.hpp"

#include <string>
#include <type_traits>

namespace specfem::test_fixture {

template <typename QuadraturePointsType> struct QuadratureRule {
  static_assert(std::is_base_of_v<QuadraturePoints::QuadraturePoints,
                                  QuadraturePointsType>,
                "QuadratureRule template parameter expects QuadraturePoints!");
  using QuadraturePoints = QuadraturePointsType;
  static constexpr int nquad = QuadraturePointsType::nquad;
  static constexpr std::array<double, nquad> quadrature_points =
      QuadraturePointsType::quadrature_points;

  static constexpr type_real evaluate_lagrange_polynomial(const int &iquad,
                                                          const type_real &x) {
    double val = 1;
    for (int i = 0; i < nquad; i++) {
      if (i != iquad) {
        val *= (x - quadrature_points[i]) /
               (quadrature_points[iquad] - quadrature_points[i]);
      }
    }
    return (type_real)val;
  }

  static std::string description(const int &indent = 0) {
    return specfem::test_fixture::impl::description<QuadraturePoints>::get(
        indent);
  }
  static std::string quadrature_name() {
    return specfem::test_fixture::impl::name<QuadraturePoints>::get();
  }
};
namespace QuadraturePoints {

struct GLL1 : QuadraturePoints {
  static constexpr int nquad = 2;
  static constexpr std::array<double, nquad> quadrature_points = { -1, 1 };

  static std::string name() { return "GLL1"; }
  static std::string description() {
    return ("2-point GLL quadrature (exactness to x^1)\n"
            "  points = [-1, 1]");
  }
};
struct GLL2 : QuadraturePoints {
  static constexpr int nquad = 3;
  static constexpr std::array<double, nquad> quadrature_points = { -1, 0, 1 };
  static std::string name() { return "GLL2"; }
  static std::string description() {
    return ("3-point GLL quadrature (exactness to x^3)\n"
            "  points = [-1, 0, 1]");
  }
};

struct Asymm5Point : QuadraturePoints {
  static constexpr int nquad = 5;
  static constexpr std::array<double, nquad> quadrature_points = { -1, -0.8,
                                                                   -0.5, 0.2,
                                                                   0.7 };
  static std::string name() { return "Asymm5"; }
  static std::string description() {
    return ("5 point asymmetric quadrature rule (low exactness interpolating "
            "quadrature for testing)\n"
            "  points = [-1, -0.8, -0.5, 0.2, 0.7]");
  }
};
struct Asymm4Point : QuadraturePoints {
  static constexpr int nquad = 4;
  static constexpr std::array<double, nquad> quadrature_points = { -0.3, 0, 0.4,
                                                                   0.6 };
  static std::string name() { return "Asymm4"; }
  static std::string description() {
    return ("4 point asymmetric quadrature rule (low exactness interpolating "
            "quadrature for testing)\n"
            "  points = [-0.3, 0, 0.4, -0.6]");
  }
};

} // namespace QuadraturePoints

} // namespace specfem::test_fixture
