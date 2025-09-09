#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

template <specfem::dimension::type DimensionTag> struct interface_types;

} // namespace specfem::assembly

#include "interface_types/dim2/interface_types.hpp"
