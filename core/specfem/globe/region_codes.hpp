#pragma once

#include "specfem/element/tags.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace specfem::globe {

namespace region_codes_impl {

// SPECFEM3D_GLOBE IREGION_* codes are one-based indices into this table.
inline constexpr std::array regions{ specfem::element::region_tag::crust_mantle,
                                     specfem::element::region_tag::outer_core,
                                     specfem::element::region_tag::inner_core };

} // namespace region_codes_impl

/** @brief Decode a region code shared by the globe database and evaluator. */
inline specfem::element::region_tag to_region_tag(const int code) {
  if (code < 1 || code > static_cast<int>(region_codes_impl::regions.size())) {
    throw std::runtime_error("Unknown globe region code " +
                             std::to_string(code));
  }
  return region_codes_impl::regions[code - 1];
}

/** @brief Encode a region tag for the globe database and evaluator. */
inline int to_region_code(const specfem::element::region_tag region) {
  for (int i = 0; i < static_cast<int>(region_codes_impl::regions.size());
       ++i) {
    if (region_codes_impl::regions[i] == region) {
      return i + 1;
    }
  }
  throw std::runtime_error("Unknown globe region tag " +
                           std::to_string(static_cast<int>(region)));
}

} // namespace specfem::globe
