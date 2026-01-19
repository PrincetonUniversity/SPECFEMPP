#pragma once

#include "Kokkos_Core.hpp"
#include "initializers.hpp"
#include "specfem_setup.hpp"

#include <string>
#include <type_traits>
#include <utility>

namespace specfem::test_fixture {

template <typename QuadraturePointsType> struct QuadratureRule {
  static_assert(std::is_base_of_v<QuadraturePoints::QuadraturePoints,
                                  QuadraturePointsType>,
                "QuadratureRule template parameter expects QuadraturePoints!");

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
    std::array<double, nquad> L = polynomial_coefficients[iquad];

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
                                                       const type_real &x) {
    double result = 0;
    double xpow = 1;
    for (int ideg = 0; ideg < nquad - 1; ++ideg) {
      result += xpow * derivative_coefficients[iquad][ideg];
      xpow *= x;
    }
    return result;
  }
};
namespace QuadraturePoints {

struct GLL1 : QuadraturePoints {
  static constexpr int nquad = 2;
  static constexpr std::array<double, nquad> quadrature_points = { -1, 1 };

  static std::string description() { return "GLL1 (-1, 1)"; }
};
struct GLL2 : QuadraturePoints {
  static constexpr int nquad = 3;
  static constexpr std::array<double, nquad> quadrature_points = { -1, 0, 1 };
  static std::string description() { return "GLL2 (-1, 0, 1)"; }
};

struct Asymm5Point : QuadraturePoints {
  static constexpr int nquad = 5;
  static constexpr std::array<double, nquad> quadrature_points = { -1, -0.8,
                                                                   -0.5, 0.2,
                                                                   0.7 };
  static std::string description() {
    return "5 point asymmetric (low exactness interpolating quadrature for "
           "testing)";
  }
};
struct Asymm4Point : QuadraturePoints {
  static constexpr int nquad = 4;
  static constexpr std::array<double, nquad> quadrature_points = { -0.3, 0, 0.4,
                                                                   0.6 };
  static std::string description() {
    return "4 point asymmetric (low exactness interpolating quadrature for "
           "testing)";
  }
};

} // namespace QuadraturePoints

} // namespace specfem::test_fixture
