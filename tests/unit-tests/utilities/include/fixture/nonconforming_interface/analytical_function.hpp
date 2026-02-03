#pragma once

#include "initializers.hpp"
#include "specfem/setup.hpp"

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
} // namespace AnalyticalFunctionType

} // namespace specfem::test_fixture
