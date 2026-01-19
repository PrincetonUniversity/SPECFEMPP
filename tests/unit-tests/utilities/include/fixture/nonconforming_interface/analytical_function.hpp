#pragma once

#include "../impl/descriptions.hpp"
#include "Serial/Kokkos_Serial_Parallel_Range.hpp"
#include "initializers.hpp"
#include "specfem_setup.hpp"

namespace specfem::test_fixture {

namespace AnalyticalFunctionType {

/**
 * @brief Describes a function f(x) = x^k for a power k
 *
 * @tparam power the exponent.
 */
template <int power> struct Power : AnalyticalFunctionType {
  static constexpr int num_components = 1;
  static std::array<type_real, num_components>
  evaluate(const type_real &coord) {
    return { (type_real)std::pow(coord, power) };
  }

  static std::string description() {
    return std::string("x^") + std::to_string(power);
  }
  static std::string name() {
    return std::string("Pow(x,") + std::to_string(power) + ")";
  }
};

/**
 * @brief Concatenates k functions into an array-valued function.
 *
 * This acts just like numpy.concatenate for rank-1 arrays or itertools.chain
 * for iterables, creating a vector-valued function, with components coming from
 * each `AnalyticalFunctions...` parameter.
 *
 * @tparam AnalyticalFunctions analytial function types to be concatenated.
 */
template <typename... AnalyticalFunctions>
struct Chain : AnalyticalFunctionType {
  static_assert(
      ((std::is_base_of_v<AnalyticalFunctionType, AnalyticalFunctions>) && ...),
      "Chain needs all of its parameters to be of "
      "AnalyticalFunctionType!");
  static constexpr int num_components =
      ((AnalyticalFunctions::num_components) + ...);
  static std::array<type_real, num_components>
  evaluate(const type_real &coord) {
    std::array<type_real, num_components> arr;
    auto it = arr.begin();
    (
        [&]() {
          const auto sub = AnalyticalFunctions::evaluate(coord);
          std::copy(sub.begin(), sub.end(), it);
          it += AnalyticalFunctions::num_components;
        }(),
        ...);
    return arr;
  }

  static std::string description() {
    return std::string("Chain (") + std::to_string(num_components) +
           " components)\n" +
           ((specfem::test_fixture::impl::description<AnalyticalFunctions>::get(
                 2) +
             "\n") +
            ...);
  }
  static std::string name() {
    return std::string("Chain(") +
           ((specfem::test_fixture::impl::name<AnalyticalFunctions>::get() +
             ",") +
            ...) +
           ")";
  }
};

/**
 * @brief Sums k analytical functions.
 *
 * Represents the pointwise-sum of two or more functions. For example,
 * `Sum<AnalyticalFunctionType::Power<1>,AnalyticalFunctionType::Power<0>>`
 * represents the function mapping `x` to `x^1 + x^0 = x + 1`.
 *
 * Each function parameter must have the same dimensions. Unlike `numpy`, we
 * have not written any support for broadcasting.
 *
 * @tparam AnalyticalFunctions analytical function types to be added together.
 */
template <typename... AnalyticalFunctions> struct Sum : AnalyticalFunctionType {
  static_assert(
      ((std::is_base_of_v<AnalyticalFunctionType, AnalyticalFunctions>) && ...),
      "Sum needs all of its parameters to be of "
      "AnalyticalFunctionType!");
  static constexpr int num_components =
      std::min((AnalyticalFunctions::num_components)...);

  static_assert(
      ((AnalyticalFunctions::num_components == num_components) && ...),
      "Sum needs all of its parameters to have the same number of components!");
  static std::array<type_real, num_components>
  evaluate(const type_real &coord) {
    std::array<type_real, num_components> arr{ 0 };
    (
        [&]() {
          const auto sub = AnalyticalFunctions::evaluate(coord);
          for (int icomp = 0; icomp < num_components; ++icomp) {
            arr[icomp] += sub[icomp];
          }
        }(),
        ...);
    return arr;
  }

  static std::string description() {
    return std::string("Sum (") + std::to_string(num_components) +
           " components)\n" +
           ((specfem::test_fixture::impl::description<AnalyticalFunctions>::get(
                 2) +
             "\n") +
            ...);
  }
  static std::string name() {
    return std::string("Sum(") +
           ((AnalyticalFunctions::description() + ",") + ...) + ")";
  }
};

} // namespace AnalyticalFunctionType

} // namespace specfem::test_fixture
