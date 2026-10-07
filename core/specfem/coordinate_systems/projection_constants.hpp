#pragma once

#include <numbers>

namespace specfem {
namespace coordinate_systems {

/// Double-precision constants for coordinate-projection math. These are
/// deliberately distinct from @ref specfem::constants::pi, which follows
/// `type_real` and may be single precision; coordinate projections always
/// compute in double precision.
constexpr double pi = std::numbers::pi;
constexpr double degrees_to_radians = pi / 180.0;
constexpr double radians_to_degrees = 180.0 / pi;

} // namespace coordinate_systems
} // namespace specfem
