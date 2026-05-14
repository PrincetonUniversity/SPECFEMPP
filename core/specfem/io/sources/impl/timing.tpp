#pragma once

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace specfem {
namespace io {

template <specfem::element::dimension_tag DimensionTag>
type_real adjust_source_timing(
    std::vector<
        std::shared_ptr<specfem::sources::source<DimensionTag> > > &sources,
    type_real user_t0) {

  const bool user_defined_start_time =
      (std::abs(user_t0) > std::numeric_limits<type_real>::epsilon());

  type_real min_t0 = std::numeric_limits<type_real>::max();
  type_real min_tshift = std::numeric_limits<type_real>::max();
  for (auto &source : sources) {
    type_real cur_t0 = source->get_t0();
    type_real cur_tshift = source->get_tshift();
    if (cur_t0 < min_t0) {
      min_t0 = cur_t0;
    }
    if (cur_tshift < min_tshift) {
      min_tshift = cur_tshift;
    }
  }

  type_real t0;
  if (user_defined_start_time) {
    if (user_t0 > min_t0 - min_tshift)
      throw std::runtime_error("User defined start time is less than minimum "
                               "required for stability");

    t0 = user_t0;
  } else {
    // Update tshift for auto detected start time
    for (auto &source : sources) {
      type_real cur_t0 = source->get_t0();
      source->update_tshift(cur_t0 - min_t0);
    }

    t0 = min_t0;
  }

  return t0;
}

template <specfem::element::dimension_tag DimensionTag>
void validate_source_simulation_type(
    const std::vector<
        std::shared_ptr<specfem::sources::source<DimensionTag> > > &sources,
    specfem::simulation::type simulation_type) {

  if (sources.empty()) {
    throw std::runtime_error("No sources found");
  }

  int number_of_adjoint_sources = 0;
  for (const auto &source : sources) {
    if (source->get_wavefield_type() ==
        specfem::simulation::field_type::adjoint) {
      number_of_adjoint_sources++;
    }
  }

  if (simulation_type == specfem::simulation::type::combined &&
      number_of_adjoint_sources == 0) {
    throw std::runtime_error("No adjoint sources found in the sources file");
  }

  if (simulation_type == specfem::simulation::type::forward &&
      number_of_adjoint_sources > 0) {
    throw std::runtime_error("Adjoint sources found in the sources file for "
                             "forward simulation");
  }
}

} // namespace io
} // namespace specfem
