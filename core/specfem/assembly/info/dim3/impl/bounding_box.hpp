#pragma once
#include "specfem/assembly/info/impl/bounds.hpp"
#include "specfem/assembly/info/impl/bounding_box.hpp"

namespace specfem::assembly::info::impl
{
  template <>
  struct BoundingBox<specfem::dimension::type::dim3> {
    Bounds x;
    Bounds y;
    Bounds z;
  };

} // namespace specfem::assembly::info::impl

