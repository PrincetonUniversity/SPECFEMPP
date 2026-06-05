#pragma once

#include "coeff_manip.hpp"
#include "specfem/enums.hpp"

#include <stdexcept>

namespace specfem::quadrature::compiletime::impl {

constexpr double fabs(const double &val) { return (val > 0) ? val : (-val); }

template <std::size_t size>
struct root_array : public std::tuple<double[size]> {

  using std::tuple<double[size]>::tuple;

  constexpr inline double &operator[](const std::size_t &i) {
    return std::get<0>(*this)[i];
  }
  constexpr const double &operator[](const std::size_t &i) const {
    return std::get<0>(*this)[i];
  }

  consteval std::array<double, size> to_array() {
    std::array<double, size> arr;
    for (int i = 0; i < size; i++) {
      arr[i] = (*this)[i];
    }
    return arr;
  }
};

/**
 * @brief Computes the newton step with deflation
 *
 * @tparam RationalPolynomialCoefficientTuple
 * @param x - the current point
 * @param px - p(x)
 * @param ppx - p'(x)
 * @param roots_found - number of roots found already.
 */
template <typename RootArrayType>
consteval double rootfind_newton_step(double x, const double &px,
                                      const double &ppx,
                                      const RootArrayType &roots_array,
                                      const int &num_roots_found) {
  double factor = 0;
  for (int i = 0; i < num_roots_found; i++) {
    factor += px / (x - roots_array[i]);
  }
  x -= px / (ppx - factor);
  return x;
}
template <typename RationalPolynomialCoefficientTuple, typename RootArrayType>
consteval double rootfind_newton_step(RationalPolynomialCoefficientTuple,
                                      const double &x, const double &px,
                                      const RootArrayType &roots_array,
                                      const int &num_roots_found) {
  return rootfind_newton_step(
      x, px,
      evaluate_polynomial(differentiate(RationalPolynomialCoefficientTuple()),
                          x),
      roots_array, num_roots_found);
}
template <typename RationalPolynomialCoefficientTuple, typename RootArrayType>
consteval double rootfind_newton_step(RationalPolynomialCoefficientTuple,
                                      const double &x,
                                      const RootArrayType &roots_array,
                                      const int &num_roots_found) {
  return rootfind_newton_step(
      x, evaluate_polynomial(RationalPolynomialCoefficientTuple(), x),
      evaluate_polynomial(differentiate(RationalPolynomialCoefficientTuple()),
                          x),
      roots_array, num_roots_found);
}

/**
 * @brief Populates polynomial roots.
 *
 *
 * # Finding Zeros of Polynomials
 *
 * This follows somewhat closely to Stoer & Bulirsch (2002) Section 5.5.
 *
 *
 * Suppose we have a degree $n > 0$ polynomial $p$, with distinct real roots
 * $\xi_1 < \dots < \xi_n$. We notice that outside of $(\xi_1,\xi_n)$, $p$ is
 * convex or concave.
 * - This is because $p'$ and $p''$ have all real roots inside $(\xi_1,\xi_n)$
 * (since they must be between the roots of $p$ and $p'$, respectively), so $p'$
 * and $p''$ cannot change sign. Without loss of generality, assume
 * $\lim_{x\to -\infty} p(x) = -\infty$. Then, if $x < \xi_1$, we know that
 * $p'(x)> 0$. If $p''(x) > 0$, then $\lim_{x\to -\infty} p''(x) > 0$ by nature
 * of polynomials (it cannot converge to zero). Hence, at some point $p'$ must
 * change sign to the left of $x$, which contradicts the assertion that $p'$ has
 * all of its roots to the right of $\xi_1$.
 *
 * Hence, starting outside of that region, Newton's method will monotonically
 * converge to the outermost root.
 *
 *
 *
 * ## Deflation
 *
 * Once we found a root $\xi$, we can repeat the process for
 * $q(x) = \frac{p(x)}{(x-\xi)}$. To avoid difficulties of stacking numerical
 * errors on $\xi$, we notice by the quotient rule:
 * $$q'(x)= \frac{(x-\xi)p'(x) - p(x)}{(x-\xi)^2}$$
 *
 * $$\frac{q(x)}{q'(x)} = \frac{q(x)(x-\xi)^2}{(x-\xi)p'(x) - p(x)} =
 * \frac{p(x)}{p'(x) - \frac{p(x)}{x-\xi}}$$
 *
 * For induction, if $p_k(x) = p(x) / \prod_{j=1}^k (x - \xi_j)$ and
 * $$\frac{p_k(x)}{p_k'(x)} = \frac{p(x)}{p'(x) - \sum_{j=1}^k
 * \frac{p(x)}{x-\xi_j}}$$
 *
 * then defining $p_{k+1}(x) = p_{k}(x) / (x - \xi_{k+1})$ yields
 *
 * $$\frac{p_{k+1}(x)}{p_{k+1}'(x)} = \frac{p_k(x)}{p_k'(x) -
 * \frac{p_k(x)}{x-\xi_{k+1}}} = \frac{1}{\frac{p'(x) - \sum_{j=1}^k
 * \frac{p(x)}{x-\xi_j}}{p(x)} - \frac{1}{x-\xi_{k+1}}}= \frac{p(x)}{p'(x) -
 * \sum_{j=1}^{k+1} \frac{p(x)}{x-\xi_j}}$$
 *
 * Use of L'Hospital's rule can be used when roots are found for the next Newton
 * step:
 *
 * $$\lim_{x\to \xi_k} \frac{p(x)}{p'(x)- \sum_{j=1}^k \frac{p(x)}{x-\xi_j}} =
 * \lim_{x\to \xi_k} \frac{p(x)(x-\xi_k)}{p'(x)(x - \xi_k) - p(x) -
 * \sum_{j=1}^{k-1} \frac{p(x)(x - \xi_k)}{x-\xi_j}}=\lim_{x\to \xi_k}
 * \frac{p(x) + p'(x)(x-\xi_k)}{p'(x) + p''(x)(x - \xi_k) - p'(x) -
 * \sum_{j=1}^{k-1} \frac{p(x) + p'(x)(x-\xi_k)}{x-\xi_j}} =
 * \frac{2p'(\xi_k)}{p''(\xi_k) - \sum_{j=1}^{k-1}
 * \frac{2p'(\xi_k)}{\xi_k-\xi_j}}$$
 *
 * Notice how this looks just like $\frac{p_{k-1}(\xi_k)}{p_{k-1}'(\xi_k)}$, but
 * with $2p'(x)$ instead of $p(x)$ and $p''(x)$ instead of $p'(x)$.
 *
 *
 * @tparam polynomial_class
 * @param poly - the polynomial class to operate on
 * @param x - the first point to check, this should be outside the
 * bounds of the roots
 * @param eps - tolerance, where |p(x)| < eps is considered a zero
 * @param max_iters - maximum number of newton iterations before a fail error.
 */
template <typename RationalPolynomialCoefficientTuple>
consteval root_array<std::tuple_size_v<RationalPolynomialCoefficientTuple> - 1>
rootfind(const RationalPolynomialCoefficientTuple &poly, double x,
         const double &eps = 1e-13, const int &max_iters = 1000) {
  const int degree = std::tuple_size_v<RationalPolynomialCoefficientTuple> - 1;
  constexpr auto pp = differentiate(RationalPolynomialCoefficientTuple());
  constexpr auto ppp = differentiate(pp);
  root_array<degree> roots;
  if (degree == 0) {
    return roots;
  }

  // first root
  double px = evaluate_polynomial(poly, x);
  int num_iters = 0;
  while (fabs(px) > eps) {
    if (num_iters > max_iters) {
      throw std::logic_error("populate_roots: Number of iterations exceeded!");
    }
    x = rootfind_newton_step(poly, x, px, roots, 0);
    px = evaluate_polynomial(poly, x);
    num_iters++;
  }

  roots[0] = x;

  // remaining roots
  int roots_found = 1;
  while (roots_found < degree) {
    // the last found root cancels the p' term, as in the notebook.
    x = rootfind_newton_step(x, 2 * evaluate_polynomial(pp, x),
                             evaluate_polynomial(ppp, x), roots,
                             roots_found - 1);

    px = evaluate_polynomial(poly, x);
    num_iters = 0;
    // take at least a few steps to ensure we moved away from the previous root
    while (num_iters < 2 || fabs(px) > eps) {
      if (num_iters > max_iters) {
        throw std::logic_error(
            "populate_roots: Number of iterations exceeded!");
      }
      x = rootfind_newton_step(poly, x, px, roots, roots_found);
      px = evaluate_polynomial(poly, x);
      num_iters++;
    }
    roots[roots_found] = x;
    roots_found++;
  }

  return roots;
}

} // namespace specfem::quadrature::compiletime::impl
