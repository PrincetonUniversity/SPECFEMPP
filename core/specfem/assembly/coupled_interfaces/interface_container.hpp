#include "enumerations/interface.hpp"

namespace specfem::assembly {

/**
 * @brief Container for coupled interfaces between two mediums
 * @tparam DimensionTag Dimension of spectral elements
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2>
struct interface_container;

} // namespace specfem::assembly

#include "dim2/impl/interface_container.hpp"
#include "dim2/impl/interface_container.tpp"
