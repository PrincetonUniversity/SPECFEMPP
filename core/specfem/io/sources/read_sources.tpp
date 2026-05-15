#pragma once

// Internal Includes
#include "specfem/io.hpp"
#include "specfem/io/sources/impl/reader.hpp"
#include "specfem/io/sources/impl/timing.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/utilities.hpp"

// External Includes
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace specfem {
namespace io {
namespace detail {

inline specfem::simulation::field_type
wavefield_type_from_simulation(specfem::simulation::type simulation_type) {
  switch (simulation_type) {
  case specfem::simulation::type::forward:
    return specfem::simulation::field_type::forward;
  case specfem::simulation::type::combined:
    return specfem::simulation::field_type::backward;
  default:
    throw std::runtime_error("Unknown simulation type");
  }
}

} // namespace detail

template <specfem::element::dimension_tag DimensionTag>
std::tuple<std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>,
           type_real>
read_sources(
    const std::vector<specfem::enums::source_file_entry> &entries,
    const int nsteps, const type_real user_t0, const type_real dt,
    const specfem::simulation::type simulation_type) {

  const auto source_wavefield_type =
      detail::wavefield_type_from_simulation(simulation_type);

  std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
      all_sources;

  for (const auto &entry : entries) {
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>> batch;

    switch (entry.format) {
    case specfem::enums::source_format::YAML:
      batch = specfem::io::sources_impl::read<
          DimensionTag, specfem::enums::source_format::YAML>(
          entry.file_path, nsteps, dt, source_wavefield_type);
      break;
    case specfem::enums::source_format::CMTSOLUTION:
      throw std::runtime_error("CMTSOLUTION reader not yet implemented");
    case specfem::enums::source_format::FORCESOLUTION:
      throw std::runtime_error("FORCESOLUTION reader not yet implemented");
    }

    all_sources.insert(all_sources.end(), batch.begin(), batch.end());
  }

  specfem::io::sources_impl::validate_source_simulation_type<DimensionTag>(
      all_sources, simulation_type);

  type_real t0 =
      specfem::io::sources_impl::adjust_source_timing<DimensionTag>(
          all_sources, user_t0);

  return std::make_tuple(all_sources, t0);
}

} // namespace io
} // namespace specfem
