#pragma once
#include "enumerations/interface.hpp"

namespace specfem::assembly::info_impl
{
  template <specfem::dimension::type DimensionTag>
  struct BoundingBox;  
} // namespace specfem::assembly::info_impl

#include "specfem/assembly/info/dim2/impl/bounding_box.hpp"
#include "specfem/assembly/info/dim3/impl/bounding_box.hpp"
