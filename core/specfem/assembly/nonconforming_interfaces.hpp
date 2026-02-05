#pragma once

#include "specfem/element_connections.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/enums.hpp"

namespace specfem::assembly {

namespace nonconforming_interfaces_impl {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_connections::type ConnectionTag>
struct interface_container;

} // namespace nonconforming_interfaces_impl

/**
 * @brief Information on coupled interfaces between two mediums
 * @tparam DimensionTag Dimension of spectral elements
 */
template <specfem::element::dimension_tag DimensionTag>
struct nonconforming_interfaces;

} // namespace specfem::assembly

#include "nonconforming_interfaces/dim2/nonconforming_interfaces.hpp"
