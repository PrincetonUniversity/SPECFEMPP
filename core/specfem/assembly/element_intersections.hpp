#pragma once

#include "specfem/element.hpp"

namespace specfem::assembly {

template <specfem::element::dimension_tag DimensionTag>
struct element_intersections;

} // namespace specfem::assembly

#include "element_intersections/element_intersections.hpp"
