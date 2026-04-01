#pragma once
#include "specfem/setup.hpp"
#include <cmath>

namespace specfem::utilities {
/**
 * @brief Compute the logarithmic center frequency between two frequencies.
 *
 * @param f1 Frequency 1
 * @param f2 Frequency 2
 * @return type_real
 */
type_real logarithmic_center(const type_real f1, const type_real f2);

} // namespace specfem::utilities
