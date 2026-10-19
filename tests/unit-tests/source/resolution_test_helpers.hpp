#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"

#include <memory>

// Shared fixtures for the source local-frame resolution tests. A named
// namespace (not an anonymous one) so the two test files can be merged into one
// unity translation unit without colliding.
namespace source_resolution_test {

inline std::unique_ptr<specfem::source_time_functions::stf> make_stf() {
  return std::make_unique<specfem::source_time_functions::Ricker>(
      10, 0.01, 1.0, 0.0, 1.0, false);
}

inline std::unique_ptr<specfem::coordinate_systems::coordinates<
    specfem::element::dimension_tag::dim3>>
origin_coords() {
  return std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>>(0.0, 0.0, 0.0);
}

} // namespace source_resolution_test
