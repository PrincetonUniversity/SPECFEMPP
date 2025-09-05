
#pragma once

#include "enumerations/interface.hpp"

namespace specfem::point {

template <specfem::dimension::type DimensionTag> class interface_index {
public:
  specfem::point::index<DimensionTag> self_index;
  specfem::point::index<DimensionTag> coupled_index;
  int ipoint;
};

} // namespace specfem::point
