#pragma once

#include "enumerations/interface_tags.hpp"

namespace specfem::assembly {

namespace conforming_interfaces_impl {

template <specfem::element::dimension_tag DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::connections::type ConnectionTag>
struct interface_container;

} // namespace conforming_interfaces_impl

/**
 * @brief Information on coupled interfaces between two mediums
 * @tparam DimensionTag Dimension of spectral elements
 */
template <specfem::element::dimension_tag DimensionTag>
struct conforming_interfaces;

} // namespace specfem::assembly

#include "conforming_interfaces/dim2/conforming_interface.hpp"
