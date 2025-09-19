#include "enumerations/interface.hpp"

namespace specfem::assembly {

namespace coupled_interfaces_impl {

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct interface_container;

} // namespace coupled_interfaces_impl

/**
 * @brief Information on coupled interfaces between two mediums
 * @tparam DimensionTag Dimension of spectral elements
 */
template <specfem::dimension::type DimensionTag> struct coupled_interfaces;

} // namespace specfem::assembly

#include "coupled_interfaces/dim2/coupled_interface.hpp"
