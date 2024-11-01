#pragma once

namespace specfem {
namespace display {

enum class format { PNG, JPG };

enum class wavefield {
  displacement_x,
  displacement_z,
  velocity_x,
  velocity_z,
  acceleration_x,
  acceleration_z,
  pressure
};
} // namespace display
} // namespace specfem
