#pragma once

#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"

#include <optional>

namespace specfem {
namespace attenuation {

/**
 * @brief Bundles all attenuation configuration needed by read_mesh.
 *
 * Construct via @c
 * specfem::runtime_configuration::setup::get_attenuation_setup().
 */
struct Setup {
  bool enabled = false;
  std::optional<specfem::units::Hertz> f0;
  specfem::utilities::Band<specfem::units::Hertz> band;
};

} // namespace attenuation
} // namespace specfem
