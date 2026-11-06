#pragma once

#include "specfem/element.hpp"
#include <map>

namespace specfem::assembly::info::impl {

/**
 * @brief Local element count per globe region.
 *
 * @tparam ElementTypes An @c element_types specialisation.
 * @param element_types Element classification of this rank.
 * @return One entry per @c region_tag; empty when @p element_types carries no
 *         globe element context (Cartesian meshes).
 */
template <typename ElementTypes>
std::map<specfem::element::region_tag, int>
count_elements_per_region(const ElementTypes &element_types) {
  std::map<specfem::element::region_tag, int> counts;
  if (!element_types.has_element_context()) {
    return counts;
  }
  for (const auto region : { specfem::element::region_tag::crust_mantle,
                             specfem::element::region_tag::outer_core,
                             specfem::element::region_tag::inner_core }) {
    counts[region] = element_types.get_number_of_elements(region);
  }
  return counts;
}

} // namespace specfem::assembly::info::impl
