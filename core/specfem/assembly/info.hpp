#pragma once

#include "enumerations/interface.hpp"
#include "specfem_setup.hpp"

namespace specfem::assembly {

  template <specfem::dimension::type DimensionTag> struct Info;

} // namespace specfem::assembly

#include "specfem/assembly/info/dim2/info.hpp"
