#include "specfem/io/sources/impl/reader.hpp"
#include "specfem/io/sources/impl/solution_format_helpers.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim3>>>
specfem::io::sources_impl::read<specfem::element::dimension_tag::dim3,
                                specfem::enums::source_format::FORCESOLUTION>(
    const std::string &file_path, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type) {

  constexpr int NLINES_PER_SOURCE = 11;
  constexpr auto dim3 = specfem::element::dimension_tag::dim3;

  auto lines = specfem::io::sources_impl::read_nonempty_lines(file_path);

  if (lines.empty() || lines.size() % NLINES_PER_SOURCE != 0) {
    throw std::runtime_error(
        "FORCESOLUTION file must contain a multiple of 11 non-empty lines, "
        "got " +
        std::to_string(lines.size()) + " in " + file_path);
  }

  const int nsources = static_cast<int>(lines.size()) / NLINES_PER_SOURCE;

  std::vector<std::shared_ptr<specfem::sources::source<dim3>>> sources;
  sources.reserve(nsources);

  for (int isrc = 0; isrc < nsources; ++isrc) {
    const int base = isrc * NLINES_PER_SOURCE;

    // Line 0: "FORCE  001" header — skip
    // Build field map from remaining 10 key:value lines
    auto fields = specfem::io::sources_impl::build_field_map(
        std::span<const std::string>(lines).subspan(base + 1, 10));

    // Time shift and half-duration / frequency
    auto tshift = specfem::io::sources_impl::get_real(fields, "time shift");
    auto hdurorf0 = specfem::io::sources_impl::get_real(fields, "hdurorf0");

    // Coordinates (x, y required; z or depth required)
    auto x = specfem::io::sources_impl::get_real(fields, "x");
    auto y = specfem::io::sources_impl::get_real(fields, "y");
    auto z =
        fields.contains("z")
            ? specfem::io::sources_impl::get_real(fields, "z")
            : -specfem::io::sources_impl::get_real(fields, "depth") * 1000.0;

    // STF type, factor, and force direction
    auto stf_type =
        specfem::io::sources_impl::get_int(fields, "source time function");
    auto factor =
        specfem::io::sources_impl::get_real(fields, "factor force source");
    auto comp_X = specfem::io::sources_impl::get_real(
        fields, "component dir vect source x");
    auto comp_Y = specfem::io::sources_impl::get_real(
        fields, "component dir vect source y");
    auto comp_Z = specfem::io::sources_impl::get_real(
        fields, "component dir vect source z_up");

    // Compute force components
    type_real fx = factor * comp_X;
    type_real fy = factor * comp_Y;
    type_real fz = factor * comp_Z;

    // Construct STF based on type
    std::unique_ptr<specfem::source_time_functions::stf> stf_ptr;
    switch (stf_type) {
    case 0:
    case 4:
      stf_ptr = std::make_unique<specfem::source_time_functions::GaussianHdur>(
          nsteps, dt, hdurorf0, tshift, 1.0, false);
      break;
    case 1:
      stf_ptr = std::make_unique<specfem::source_time_functions::Ricker>(
          nsteps, dt, hdurorf0, tshift, 1.0, false);
      break;
    case 2:
      stf_ptr = std::make_unique<specfem::source_time_functions::Heaviside>(
          nsteps, dt, hdurorf0, tshift, 1.0, false);
      break;
    default:
      throw std::runtime_error(
          "Unsupported source time function type in FORCESOLUTION: " +
          std::to_string(stf_type));
    }

    // Construct source (no datetime for FORCESOLUTION)
    sources.push_back(std::make_shared<specfem::sources::force<dim3>>(
        x, y, z, fx, fy, fz, std::move(stf_ptr), wavefield_type));
  }

  return sources;
}
