#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly::boundary_values_impl {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::boundary_tag BoundaryTag>
class boundary_medium_container;

template <specfem::dimension::type DimensionTag,
          specfem::element::boundary_tag BoundaryTag>
class boundary_value_container;

} // namespace specfem::assembly::boundary_values_impl

#include "boundary_values/dim2/impl/boundary_medium_container.hpp"
#include "boundary_values/dim2/impl/boundary_value_container.hpp"

#include "boundary_values/boundary_values.hpp"
