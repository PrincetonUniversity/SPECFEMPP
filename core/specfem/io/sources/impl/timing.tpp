#pragma once

#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

template <specfem::element::dimension_tag DimensionTag>
std::tuple<type_real, std::optional<specfem::datetime::type>>
specfem::io::sources_impl::adjust_source_timing(
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
        &sources,
    type_real user_t0) {

  // Helper to convert type_real seconds to chrono milliseconds
  auto to_ms = [](type_real seconds) {
    return std::chrono::round<std::chrono::milliseconds>(
        std::chrono::duration<double>(static_cast<double>(seconds)));
  };

  const bool user_defined_start_time =
      (std::abs(user_t0) > std::numeric_limits<type_real>::epsilon());

  // Count sources with starttimes before any tshift modification
  int sources_with_starttime = 0;
  for (const auto &source : sources) {
    if (source->get_starttime())
      sources_with_starttime++;
  }

  // Validate starttime consistency early
  const bool all_have_starttime =
      (sources_with_starttime == static_cast<int>(sources.size()));
  if (sources_with_starttime > 1 && !all_have_starttime) {
    throw std::runtime_error(
        "Inconsistent source starttimes: " +
        std::to_string(sources_with_starttime) + " of " +
        std::to_string(sources.size()) +
        " sources have a starttime. Either all or exactly one must.");
  }

  // Compute min t0 (STF stability time) across all sources
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
    t0 = min_t0;

    // Only adjust tshifts from STF t0 values when absolute starttimes
    // won't override them. When all sources have starttimes, the
    // multi-starttime branch below computes tshifts from absolute times.
    if (!all_have_starttime) {
      for (auto &source : sources) {
        type_real cur_t0 = source->get_t0();
        source->update_tshift(cur_t0 - min_t0);
      }
    }
  }

  // Compute simulation start datetime from sources that carry one.
  std::optional<specfem::datetime::type> starttime;

  if (sources_with_starttime == 1) {
    // Single starttime: tshifts were adjusted above from STF t0 values.
    // Compute simulation start: UTC(t=t0) = origin_time - tshift + t0
    for (const auto &source : sources) {
      if (auto otime = source->get_starttime()) {
        starttime = *otime - to_ms(source->get_tshift()) + to_ms(t0);
        break;
      }
    }
  } else if (all_have_starttime) {
    // All sources have starttimes.
    // Find the earliest UTC(t=0) = origin_time_i - tshift_i
    // (tshifts here are the original values from the source definitions)
    specfem::datetime::type earliest_t0_utc = specfem::datetime::type::max();
    for (const auto &source : sources) {
      auto otime = *source->get_starttime();
      auto candidate = otime - to_ms(source->get_tshift());
      if (candidate < earliest_t0_utc) {
        earliest_t0_utc = candidate;
      }
    }

    // Adjust tshifts so each source fires at the correct simulation time.
    // new_tshift = (origin_time - earliest_t0_utc) in seconds
    for (auto &source : sources) {
      auto otime = *source->get_starttime();
      auto delta = otime - earliest_t0_utc; // milliseconds duration
      type_real new_tshift =
          static_cast<type_real>(
              std::chrono::duration_cast<std::chrono::microseconds>(delta)
                  .count()) /
          static_cast<type_real>(1.0e6);
      source->update_tshift(new_tshift);
    }

    // Simulation start: UTC(t=t0) = earliest_t0_utc + t0
    starttime = earliest_t0_utc + to_ms(t0);
  }
  // else: no starttimes → starttime remains nullopt

  return { t0, starttime };
}

template <specfem::element::dimension_tag DimensionTag>
void specfem::io::sources_impl::validate_source_simulation_type(
    const std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
        &sources,
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

  if ((simulation_type == specfem::simulation::type::combined ||
       simulation_type == specfem::simulation::type::combined_undoatt) &&
      number_of_adjoint_sources == 0) {
    throw std::runtime_error("No adjoint sources found in the sources file");
  }

  if (simulation_type == specfem::simulation::type::forward &&
      number_of_adjoint_sources > 0) {
    throw std::runtime_error("Adjoint sources found in the sources file for "
                             "forward simulation");
  }
}
