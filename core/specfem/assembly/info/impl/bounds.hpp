#pragma once

#include "specfem/setup.hpp"

namespace specfem::assembly::info::impl {

/**
 * @brief Min/max range for a scalar quantity.
 *
 * Stores the minimum and maximum values observed for a property
 * (e.g., velocity, density, distance). Provides utility methods
 * for computing derived quantities.
 */
struct Bounds {
public:
  type_real min; ///< Minimum value
  type_real max; ///< Maximum value

  /** @brief Default constructor initializes to zero bounds. */
  KOKKOS_INLINE_FUNCTION Bounds() : min(0), max(0) {}

  /**
   * @brief Construct bounds from explicit min/max values.
   * @param min_in Minimum value
   * @param max_in Maximum value
   */
  KOKKOS_INLINE_FUNCTION Bounds(type_real min_in, type_real max_in)
      : min(min_in), max(max_in) {}

  /** @brief Compute the range (max - min). */
  KOKKOS_INLINE_FUNCTION type_real length() const {
    return this->max - this->min;
  }

  /**
   * @brief Compute the ratio (max / min).
   * @throws std::runtime_error if min is zero
   */
  type_real ratio() const {
    if (this->min == 0) {
      throw std::runtime_error(
          "Bounds::ratio(): min is zero, cannot compute ratio.");
    }
    return this->max / this->min;
  }

  /** @brief Compute the midpoint of the range. */
  KOKKOS_INLINE_FUNCTION type_real center() const {
    return 0.5 * (this->max + this->min);
  }

  /**
   * @brief Set both min and max to the same value.
   * @param value Value to assign to both bounds
   */
  KOKKOS_INLINE_FUNCTION Bounds &operator=(const type_real value) {
    this->min = value;
    this->max = value;
    return *this;
  }
};

} // namespace specfem::assembly::info::impl
