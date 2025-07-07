#include "enumerations/interface.hpp"

namespace specfem::assembly {

/**
 * @brief Information on coupled interfaces between two mediums
 * @tparam DimensionTag Dimension of spectral elements
 */
template <specfem::dimension::type DimensionTag> struct coupled_interfaces;

} // namespace specfem::assembly

#include "coupled_interfaces/dim2/coupled_interfaces.hpp"
#include "coupled_interfaces/dim2/coupled_interfaces.tpp"
