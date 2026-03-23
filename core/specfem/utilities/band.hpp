#pragma once
#include "specfem/units/conversions.hpp"
#include "specfem/units/quantity.hpp"

namespace specfem {

namespace utilities {

/**
 * @brief Represents a spectral band with minimum and maximum values.
 *
 * This struct defines a frequency, period, or angular frequency band. Values
 * are automatically ordered so that min <= max. Only enabled for spectral
 * quantity types (Seconds, Hertz, Omega).
 *
 * @tparam T Unit type (specfem::units::Seconds, specfem::units::Hertz, or
 *           specfem::units::Omega)
 */
template <typename T,
          std::enable_if_t<std::is_same<T, specfem::units::Seconds>::value ||
                               std::is_same<T, specfem::units::Hertz>::value ||
                               std::is_same<T, specfem::units::Omega>::value,
                           int> = 0>
struct Band {
  T min; ///< Minimum value of the band
  T max; ///< Maximum value of the band

  /**
   * @brief Construct a spectral band.
   *
   * @param a First boundary value
   * @param b Second boundary value (order doesn't matter)
   */
  Band(T a, T b) : min(a <= b ? a : b), max(a <= b ? b : a) {}
};

} // namespace utilities

namespace units {

/**
 * @brief Convert a band from one unit to another.
 *
 * Operator overlead for the underlying datatypes.
 *
 * Performs unit conversion on both min and max values of the band.
 *
 * @tparam To Target unit type
 * @tparam From Source unit type
 * @param b Band to convert
 * @return Band<To> Converted band in target units
 */
template <typename To, typename From>
utilities::Band<To> unit_cast(const utilities::Band<From> &b) {
  return { specfem::units::unit_cast<To>(b.min),
           specfem::units::unit_cast<To>(b.max) };
}

} // namespace units

} // namespace specfem
