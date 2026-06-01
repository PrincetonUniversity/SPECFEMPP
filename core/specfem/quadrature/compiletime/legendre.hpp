#pragma once

#include "rational_polynomial.hpp"

namespace specfem {
namespace quadrature {
namespace compiletime {

template <int order>
struct LegendrePolynomial
    : public decltype(impl::get_rational_poly(
          typename LegendrePolynomial<order - 1>::
              template times_x_times_constant<
                  std::ratio<2 * order - 1, order>>::
                  template minus<typename LegendrePolynomial<
                      order - 2>::template times_constant<std::ratio<order - 1,
                                                                     order>>>::
                      coefficients())){};

template <>
struct LegendrePolynomial<0> : public RationalPolynomial<std::ratio<1>> {};

template <>
struct LegendrePolynomial<1>
    : public RationalPolynomial<std::ratio<0>, std::ratio<1>> {};

} // namespace compiletime
} // namespace quadrature
} // namespace specfem
