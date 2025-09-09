#include "enumerations/interface.hpp"

namespace specfem::assembly {

namespace coupled_interfaces2_impl {

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct interface_container;

} // namespace coupled_interfaces2_impl

/**
 * @brief Information on coupled interfaces between two mediums
 * @tparam DimensionTag Dimension of spectral elements
 */
template <specfem::dimension::type DimensionTag> struct coupled_interfaces2;

} // namespace specfem::assembly

#include "coupled_interfaces2/dim2/coupled_interface.hpp"
