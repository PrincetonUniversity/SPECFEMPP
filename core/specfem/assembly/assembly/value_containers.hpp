#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly::impl {
/**
 * @brief Values for every quadrature point in the finite element mesh
 *
 */
template <specfem::dimension::type,
          template <specfem::dimension::type, specfem::element::medium_tag,
                    specfem::element::property_tag> class containers_type>
struct value_containers;

} // namespace specfem::assembly::impl

#include "value_containers/dim2/value_containers.hpp"
#include "value_containers/dim3/value_containers.hpp"
