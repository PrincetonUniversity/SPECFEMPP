#pragma once

#include <ratio>
#include <tuple>
#include <utility>

namespace specfem ::quadrature::compiletime::impl {

template <typename... coeffs>
constexpr double coefficient_array[sizeof...(coeffs)] = {
  ((double)coeffs::num) / ((double)coeffs::den)...
};

template <typename... coeffs>
consteval double evaluate_polynomial(const std::tuple<coeffs...> &,
                                     const double &x) {
  double val = 0;
  double xpow = 1;
  for (std::size_t i = 0; i < sizeof...(coeffs); i++) {
    val += xpow * coefficient_array<coeffs...>[i];
    xpow *= x;
  }
  return val;
}

template <std::size_t ind, typename... coeffs>
consteval std::ratio_multiply<std::tuple_element_t<ind, std::tuple<coeffs...>>,
                              std::ratio<ind>>
component_mult_by_index(const std::tuple<coeffs...> &) {
  return {};
}

template <typename coeffs_tuple, std::size_t... Is>
consteval std::tuple<
    decltype(component_mult_by_index<Is + 1>(coeffs_tuple()))...>
differentiate(const coeffs_tuple &,
              const std::integer_sequence<std::size_t, Is...> &) {
  return {};
}

// exit condition, just in case :)
consteval std::tuple<> differentiate(const std::tuple<> &);

template <typename... coeffs>
consteval decltype(differentiate(
    std::tuple<coeffs...>(),
    std::make_integer_sequence<std::size_t, sizeof...(coeffs) - 1>()))
differentiate(const std::tuple<coeffs...> &) {
  return {};
}

template <std::size_t padded_size, typename... coeffs, std::size_t... counting>
consteval std::tuple<
    coeffs..., std::ratio_multiply<std::ratio<counting>, std::ratio<0>>...>
pad_zeros_to_size(const std::tuple<coeffs...> &,
                  const std::integer_sequence<std::size_t, counting...> &) {
  return {};
}

template <std::size_t padded_size, typename... coeffs>
consteval decltype(pad_zeros_to_size<padded_size>(
    std::tuple<coeffs...>(),
    std::make_integer_sequence<std::size_t, padded_size - sizeof...(coeffs)>()))
pad_zeros_to_size(const std::tuple<coeffs...> &) {
  return {};
}

template <typename... poly1, typename... poly2>
consteval std::enable_if_t<sizeof...(poly1) == sizeof...(poly2),
                           std::tuple<std::ratio_subtract<poly1, poly2>...>>
subtract(const std::tuple<poly1...> &, const std::tuple<poly2...> &) {}

template <typename T> consteval T max(const T &t1, const T &t2) {
  return t1 > t2 ? t1 : t2;
}

template <typename... poly1, typename... poly2>
consteval std::enable_if_t<
    sizeof...(poly1) != sizeof...(poly2),
    decltype(subtract(
        pad_zeros_to_size<max(sizeof...(poly1), sizeof...(poly2))>(
            std::tuple<poly1...>()),
        pad_zeros_to_size<max(sizeof...(poly1), sizeof...(poly2))>(
            std::tuple<poly2...>())))>
subtract(std::tuple<poly1...>, std::tuple<poly2...>) {
  return {};
}

template <typename factor, typename... coeffs>
consteval std::tuple<std::ratio_multiply<coeffs, factor>...>
const_multiply(const std::tuple<coeffs...> &) {
  return {};
}

template <typename... coeffs>
consteval std::tuple<std::ratio<0>, coeffs...>
times_x(const std::tuple<coeffs...> &) {
  return {};
}

} // namespace specfem::quadrature::compiletime::impl
