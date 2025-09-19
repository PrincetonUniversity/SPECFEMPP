#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

template <specfem::dimension::type DimensionTag> struct edge_types;

} // namespace specfem::assembly

#include "edge_types/dim2/edge_types.hpp"
