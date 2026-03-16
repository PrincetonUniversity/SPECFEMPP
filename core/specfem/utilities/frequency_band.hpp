#pragma once
#include "specfem/units/conversions.hpp"
#include "specfem/units/quantity.hpp"

namespace specfem::utilities {

template <typename T> struct FrequencyBand {
  T min, max;

  FrequencyBand(T a, T b) : min(a <= b ? a : b), max(a <= b ? b : a) {}
};

// Explicit cross-unit conversion, mirroring unit_cast
template <typename To, typename From>
FrequencyBand<To> unit_cast(const FrequencyBand<From> &b) {
  return { specfem::units::unit_cast<To>(b.min),
           specfem::units::unit_cast<To>(b.max) };
}

} // namespace specfem::utilities
