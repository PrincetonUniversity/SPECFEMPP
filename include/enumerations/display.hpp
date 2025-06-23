#pragma once

namespace specfem {
namespace display {

enum class format { PNG, JPG, on_screen };

enum class wavefield {
  displacement,
  velocity,
  acceleration,
  pressure,
  rotation
};
} // namespace display
} // namespace specfem
