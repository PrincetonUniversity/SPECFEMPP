#pragma once

#include "specfem/assembly/info/impl/bounding_box.hpp"

namespace specfem::assembly::info::impl {
  template <>
  struct BoundingBox<specfem::dimension::type::dim2> {
    Bounds x;
    Bounds z;

    BoundingBox()
        : x(0, 0),
          z(0, 0) {}

    BoundingBox(type_real x_min, type_real x_max,
                 type_real z_min, type_real z_max)
        : x(x_min, x_max),
          z(z_min, z_max) {}
  };
} // namespace specfem::assembly::info::impl  