// Internal Includes
#include "specfem/io.hpp"
#include "specfem/io/sources/impl/timing.hpp"
#include "specfem/io/sources/impl/yaml_reader.hpp"
#include "specfem/runtime_configuration/sources.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"

// External Includes
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {
specfem::simulation::field_type
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
} // namespace

// read_sources(vector<source_file_entry>, ...) — multi-format dispatch
template <>
std::tuple<std::vector<std::shared_ptr<specfem::sources::source<
               specfem::element::dimension_tag::dim2>>>,
           type_real>
specfem::io::read_sources<specfem::element::dimension_tag::dim2>(
    const std::vector<specfem::runtime_configuration::source_file_entry>
        &entries,
    const int nsteps, const type_real user_t0, const type_real dt,
    const specfem::simulation::type simulation_type) {

  const auto source_wavefield_type =
      wavefield_type_from_simulation(simulation_type);

  std::vector<std::shared_ptr<
      specfem::sources::source<specfem::element::dimension_tag::dim2>>>
      all_sources;

  for (const auto &entry : entries) {
    std::vector<std::shared_ptr<
        specfem::sources::source<specfem::element::dimension_tag::dim2>>>
        batch;

    switch (entry.format) {
    case specfem::runtime_configuration::source_format::YAML:
      batch =
          specfem::io::read_yaml_sources<specfem::element::dimension_tag::dim2>(
              entry.file_path, nsteps, dt, source_wavefield_type);
      break;
    case specfem::runtime_configuration::source_format::CMTSOLUTION:
      throw std::runtime_error("CMTSOLUTION reader not yet implemented");
    case specfem::runtime_configuration::source_format::FORCESOLUTION:
      throw std::runtime_error("FORCESOLUTION reader not yet implemented");
    }

    all_sources.insert(all_sources.end(), batch.begin(), batch.end());
  }

  specfem::io::validate_source_simulation_type<
      specfem::element::dimension_tag::dim2>(all_sources, simulation_type);

  type_real t0 =
      specfem::io::adjust_source_timing<specfem::element::dimension_tag::dim2>(
          all_sources, user_t0);

  return std::make_tuple(all_sources, t0);
}
