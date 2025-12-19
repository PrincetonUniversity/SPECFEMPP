#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

namespace specfem::test::fixture {

namespace AnalyticalFunctionType1D {

/**
 * @brief Describes a function f(x) = x^k for a power k
 *
 * @tparam power the exponent.
 */
template <int power> struct Power : AnalyticalFunctionType1D {
  static constexpr int num_components = 1;
  static type_real evaluate(const type_real &coord) {
    return std::pow(coord, power);
  }

  static std::string description() {
    return std::string("xi^") + std::to_string(power);
  }
};
} // namespace AnalyticalFunctionType1D

} // namespace specfem::test::fixture
