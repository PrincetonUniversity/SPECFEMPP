#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"

namespace specfem::utilities {

template <typename T> struct FrequencyBand {
  T min, max;
};

} // namespace specfem::utilities
