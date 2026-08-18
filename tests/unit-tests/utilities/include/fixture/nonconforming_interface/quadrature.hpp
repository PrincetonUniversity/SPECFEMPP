#pragma once

#include "../impl/descriptions.hpp"
#include "initializers.hpp"
#include "specfem/setup.hpp"

#include <string>
#include <type_traits>
#include <utility>

namespace specfem::test_fixture {

template <typename QuadraturePointsType> struct QuadratureRule {
  static_assert(std::is_base_of_v<QuadraturePoints::QuadraturePoints,
                                  QuadraturePointsType>,
                "QuadratureRule template parameter expects QuadraturePoints!");
  using QuadraturePoints = QuadraturePointsType;
  static constexpr int nquad = QuadraturePointsType::nquad;
  static constexpr std::array<double, nquad> quadrature_points =
      QuadraturePointsType::quadrature_points;

  /**
   * @brief Evaluates the `iquad`th Lagrange basis polynomial at the point `x`.
   *
   * @param iquad - the index of the polynomial (will evaluate to 1 at this knot
   * and zero everywhere else)
   * @param x - coordinate to evaluate at
   * @return double the value L[iquad] (x).
   */
  static constexpr double evaluate_lagrange_polynomial(const int &iquad,
                                                       const double &x) {

    // TODO: should we switch this to use the polynomial_coefficients?
    double val = 1;
    for (int i = 0; i < nquad; ++i) {
      if (i != iquad) {
        val *= (x - quadrature_points[i]) /
               (quadrature_points[iquad] - quadrature_points[i]);
      }
    }
    return val;
  }

  /**
   * @brief Provides the corresponding lagrange interpolation polynomial over
   * the basis {x^k}, that is, $L_{iquad} = \sum_k
   * computelagrangepolynomialcoefficients(iquad)[k] * x^k$
   *
   * @param iquad index of the lagrange interpolating polynomial to find
   * @return std::array<double, nquad> the array of coefficients
   */
  static constexpr std::array<double, nquad>
  compute_lagrange_polynomial_coefficients(const int &iquad) {
    std::array<double, nquad> coeffs{ 0 };
    coeffs[0] = 1;

    for (int i = 0; i < nquad; ++i) {
      if (i != iquad) {
        // coeffs *= (x - quadrature_points[i])/(quadrature_points[iquad] -
        // quadrature_points[i])
        double factor = 1 / (quadrature_points[iquad] - quadrature_points[i]);
        coeffs[nquad - 1] = coeffs[nquad - 2] * factor;
        for (int j = nquad - 2; j > 0; --j) {
          coeffs[j] =
              (coeffs[j - 1] - coeffs[j] * quadrature_points[i]) * factor;
        }
        coeffs[0] *= -quadrature_points[i] * factor;
      }
    }
    return coeffs;
  }

  static constexpr std::array<std::array<double, nquad>, nquad>
  _compute_all_lagrange_polynomial_coefficients() {
    std::array<std::array<double, nquad>, nquad> coeffs{};
    for (int i = 0; i < nquad; ++i) {
      coeffs[i] = compute_lagrange_polynomial_coefficients(i);
    }
    return coeffs;
  }

  static constexpr std::array<std::array<double, nquad>, nquad>
      polynomial_coefficients = _compute_all_lagrange_polynomial_coefficients();

  static constexpr double
  compute_lagrange_quadrature_weight(const int &iquad,
                                     const double &integral_start = -1,
                                     const double &integral_end = 1) {
    const std::array<double, nquad> &L = polynomial_coefficients[iquad];

    // evaluate integral at integral_end and integral_start
    double result = 0;

    double start_pow = integral_start;
    double end_pow = integral_end;
    for (int i = 0; i < nquad; ++i) {
      // add L_integral[k] * integral_end^k - L_integral[k] * integral_start^k
      result += (L[i] / (i + 1)) * (end_pow - start_pow);
      start_pow *= integral_start;
      end_pow *= integral_end;
    }

    return result;
  }

  static constexpr std::array<std::array<double, nquad - 1>, nquad>
  _compute_all_lagrange_polynomial_derivative_coefficients() {
    std::array<std::array<double, nquad - 1>, nquad> coeffs{};
    for (int ipoly = 0; ipoly < nquad; ++ipoly) {
      const auto &L = polynomial_coefficients[ipoly];
      for (int ideg = 1; ideg < nquad; ++ideg) {
        coeffs[ipoly][ideg - 1] = ideg * L[ideg];
      }
    }
    return coeffs;
  }
  static constexpr std::array<std::array<double, nquad - 1>, nquad>
      derivative_coefficients =
          _compute_all_lagrange_polynomial_derivative_coefficients();

  /**
   * @brief Provides L'(x) for the `iquad`th Lagrange polynomial.
   *
   * Evaluates the derivative of the `iquad`th Lagrange basis polynomial at the
   * point `x`.
   *
   * @param iquad - the index of the polynomial (L will evaluate to 1 at this
   * knot and zero everywhere else)
   * @param x - coordinate to evaluate at
   * @return double the value L'[iquad] (x).
   */
  static constexpr double evaluate_lagrange_derivative(const int &iquad,
                                                       const double &x) {
    double result = 0;
    double xpow = 1;
    for (int ideg = 0; ideg < nquad - 1; ++ideg) {
      result += xpow * derivative_coefficients[iquad][ideg];
      xpow *= x;
    }
    return result;
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

struct GLL4 : QuadraturePoints {
  static constexpr int nquad = 5;

  // computed from scipy.special.roots_jacobi(3,1,1)
  // (https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_jacobi.html)
  // N=5 ; print(-1,*(f"{x:.40}" for x in
  // scipy.special.roots_jacobi(N-2,1,1)[0]),1, sep=", ")

  static constexpr std::array<double, nquad> quadrature_points = {
    -1, -0.6546536707079770867068191364523954689503, 0.0,
    0.6546536707079770867068191364523954689503, 1
  };
  static std::string name() { return "GLL4"; }
  static std::string description() {
    return "5-Point (degree-4) GLL (7 degrees of exactness)";
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
struct GLL6 : QuadraturePoints {
  static constexpr int nquad = 7;

  // computed from scipy.special.roots_jacobi(5,1,1)
  // (https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_jacobi.html)
  // N=7 ; print(-1,*(f"{x:.40}" for x in
  // scipy.special.roots_jacobi(N-2,1,1)[0]),1, sep=", ")

  static constexpr std::array<double, nquad> quadrature_points = {
    -1,
    -0.8302238962785670750577082799281924962997,
    -0.4688487934707141757684212279855273663998,
    0.0,
    0.4688487934707141757684212279855273663998,
    0.8302238962785670750577082799281924962997,
    1
  };
  static std::string name() { return "GLL6"; }
  static std::string description() {
    return "7 Point (degree-6) GLL (11 degrees of exactness)";
  }
};
struct GL6 : QuadraturePoints {
  static constexpr int nquad = 7;

  // computed from scipy.special.roots_legendre(7)
  // (https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html)
  // print(*(f"{x:.40}" for x in scipy.special.roots_legendre(7)[0]), sep=", ")
  static constexpr std::array<double, nquad> quadrature_points = {
    -0.9491079123427583752459213428664952516556,
    -0.7415311855993944600839995473506860435009,
    -0.4058451513773971841558818596240598708391,
    0.0,
    0.4058451513773971841558818596240598708391,
    0.7415311855993944600839995473506860435009,
    0.9491079123427583752459213428664952516556
  };
  static std::string name() { return "GL6"; }
  static std::string description() {
    return "7 Point (degree-6) Gauss-Legendre (13 degrees of exactness)";
  }
};

} // namespace QuadraturePoints

} // namespace specfem::test_fixture
