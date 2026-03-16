#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"

namespace specfem::utilities {

template <typename T> struct FrequencyBand {
  T min, max;

  template <typename U>
  FrequencyBand(const U min_in, const U max_in)
      : min(static_cast<T>(min_in)), max(static_cast<T>(max_in)) {}

  template <typename U> explicit inline operator FrequencyBand<U>() const {
    return FrequencyBand<U>{ static_cast<U>(min), static_cast<U>(max) };
  }
};

} // namespace specfem::utilities
