#pragma once

#include "impl/coeff_manip.hpp"
#include "impl/rootfind.hpp"
#include <cstddef>
#include <tuple>
#include <utility>

namespace specfem::quadrature::compiletime {

template <typename... Coefficients> struct RationalPolynomial;

namespace impl {
// needed for type-aliasing poly when RationalPolynomial still incomplete.
template <typename... Coefficients>
constexpr RationalPolynomial<Coefficients...>
get_rational_poly(const std::tuple<Coefficients...> &) {
  return {};
}
} // namespace impl

/**
 * @brief Polynomial with rational coefficients
 *
 * @tparam coefficients the polynomial coefficients from smallest order
 * (constant term) to largest
 */
template <typename... Coefficients> struct RationalPolynomial {

  using coefficients = std::tuple<Coefficients...>;
  static constexpr std::size_t degree = sizeof...(Coefficients) - 1;
  static constexpr double coefficient_array[degree + 1] =
      impl::coefficient_array<Coefficients...>;

  /**
   * @brief numerical (double) evaluation, used for root finding.
   */
  constexpr static double evaluate(const double &x) {
    return impl::evaluate_polynomial(coefficients(), x);
  }

  constexpr RationalPolynomial(const coefficients &) {};
  constexpr RationalPolynomial() = default;

  using derivative =
      decltype(impl::get_rational_poly(impl::differentiate(coefficients())));

  template <std::size_t size>
  using zero_padded_to = decltype(impl::get_rational_poly(
      impl::pad_zeros_to_size<size>(coefficients())));

  template <typename OtherPoly>
  using minus = decltype(impl::get_rational_poly(
      impl::subtract(coefficients(), typename OtherPoly::coefficients())));

  template <typename factor>
  using times_constant = decltype(impl::get_rational_poly(
      impl::const_multiply<factor>(coefficients())));

  template <typename factor>
  using times_x_times_constant = decltype(impl::get_rational_poly(
      impl::times_x(impl::const_multiply<factor>(coefficients()))));
};

namespace impl {
/**
 * @brief returns an upper bound on the magnitude of the roots, as given by
 * Cauchy's bound.
 *
 * This is actually an overestimate, since we are including |a[n]/a[n]| = 1 in
 * the max.
 */
template <typename... coeffs>
constexpr double cauchy_bound(const std::tuple<coeffs...> &) {
  using leading_coeff =
      std::tuple_element_t<sizeof...(coeffs) - 1, std::tuple<coeffs...>>;
  double leading_coeff_d =
      ((double)leading_coeff::num) / ((double)leading_coeff::den);
  double v = 0;
  (
      [&]() {
        double coeff = ((double)coeffs::num) / ((double)coeffs::den);
        v = max(v, fabs(coeff / leading_coeff_d));
      }(),
      ...);
  return v + 1;
}
} // namespace impl

/**
 * @brief RationalPolynomial, but we also compute roots
 */
template <typename... Coefficients>
struct RationalPolynomialWithRoots
    : public RationalPolynomial<Coefficients...> {
  using RationalPolynomial<Coefficients...>::degree;
  using coefficients = std::tuple<Coefficients...>;
  static constexpr std::array<double, degree> roots =
      impl::rootfind(coefficients(), -impl::cauchy_bound(coefficients()))
          .to_array();

  constexpr RationalPolynomialWithRoots(const std::tuple<Coefficients...> &) {};
  constexpr RationalPolynomialWithRoots(
      const RationalPolynomial<Coefficients...> &) {};
  constexpr RationalPolynomialWithRoots() = default;
};

} // namespace specfem::quadrature::compiletime
