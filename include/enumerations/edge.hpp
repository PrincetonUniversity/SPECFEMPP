#pragma once

namespace specfem::connections {

enum class type : int { strongly_conforming = 1 };

enum class orientation : int {
  top,
  right,
  bottom,
  left,
  top_left,
  top_right,
  bottom_right,
  bottom_left
};

} // namespace specfem::connections
