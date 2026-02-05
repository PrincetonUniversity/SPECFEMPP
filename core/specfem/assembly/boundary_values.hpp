#pragma once

#include "enumerations/interface_tags.hpp"

namespace specfem::assembly::boundary_values_impl {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::boundary_tag BoundaryTag>
class boundary_medium_container;

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::boundary_tag BoundaryTag>
class boundary_value_container;

} // namespace specfem::assembly::boundary_values_impl

#include "boundary_values/dim2/impl/boundary_medium_container.hpp"
#include "boundary_values/dim2/impl/boundary_value_container.hpp"

#include "boundary_values/boundary_values.hpp"
