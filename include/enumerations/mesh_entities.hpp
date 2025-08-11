#pragma once

namespace specfem::mesh_entity {

enum class type : int {
  bottom = 1,
  top = 2,
  left = 3,
  right = 4,
  bottom_left = 5,
  bottom_right = 6,
  top_right = 7,
  top_left = 8
};

} // namespace specfem::mesh_entity
