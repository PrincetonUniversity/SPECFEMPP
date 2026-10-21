#pragma once

#include "specfem/plugin/injection_abi.h"
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem {
namespace injection {

/** @brief Every medium an injection point can belong to. */
inline constexpr std::array<spfp_medium, 2> media = { spfp_medium_elastic,
                                                      spfp_medium_acoustic };

/**
 * @brief Number of buffer components a quantity occupies.
 *
 * @param quantity Requested quantity
 * @return 1 for @c spfp_quantity_potential_rate, 3 for vector quantities
 * @throws std::runtime_error for an unknown quantity
 */
[[nodiscard]] constexpr int component_count(const spfp_quantity quantity) {
  switch (quantity) {
  case spfp_quantity_displacement:
  case spfp_quantity_velocity:
  case spfp_quantity_acceleration:
  case spfp_quantity_traction:
    return 3;
  case spfp_quantity_potential_rate:
    return 1;
  default:
    throw std::runtime_error("Unknown injection quantity " +
                             std::to_string(quantity) + ".");
  }
}

/**
 * @brief Number of buffer components a list of quantities occupies.
 *
 * @param quantities Requested quantities
 * @return Sum of their component counts
 */
[[nodiscard]] constexpr int
count_components(const std::span<const spfp_quantity> quantities) {
  return std::transform_reduce(quantities.begin(), quantities.end(), 0,
                               std::plus<>{}, component_count);
}

/**
 * @brief Quantities requested from a plugin, per medium.
 *
 * Defines the slot layout of the incident-field buffer: for a point of a given
 * medium, its quantities follow in request order, each expanded to its
 * component count.
 */
struct Request {
  std::vector<spfp_quantity> elastic;  ///< Quantities for elastic points
  std::vector<spfp_quantity> acoustic; ///< Quantities for acoustic points

  /**
   * @brief Quantities requested for a medium.
   * @param medium Medium of the point
   * @return The medium's quantities, in slot order
   * @throws std::runtime_error for an unknown medium
   */
  [[nodiscard]] const std::vector<spfp_quantity> &
  quantities(const spfp_medium medium) const;

  /**
   * @brief Number of slots used by a point of a medium.
   * @param medium Medium of the point
   * @return Sum of the component counts of the medium's quantities
   */
  [[nodiscard]] int components(const spfp_medium medium) const;

  /**
   * @brief First slot of a requested quantity.
   * @param medium Medium of the point
   * @param index Position of the quantity in quantities(medium)
   * @return Slot of the quantity's first component
   * @throws std::out_of_range if @p index is not a valid position
   */
  [[nodiscard]] int slot(const spfp_medium medium,
                         const std::size_t index) const;

  /**
   * @brief Number of slots per row of the buffer.
   * @return The largest components() over all media
   */
  [[nodiscard]] int max_components() const;
};

} // namespace injection
} // namespace specfem
