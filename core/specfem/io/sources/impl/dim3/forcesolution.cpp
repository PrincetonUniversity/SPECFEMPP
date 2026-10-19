#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
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
    // Half-duration / Ricker f0. SPECFEM3D_GLOBE labels this "half duration";
    // regional SPECFEM3D uses "hdurorf0". Accept either.
    auto hdurorf0 =
        fields.contains("half duration")
            ? specfem::io::sources_impl::get_real(fields, "half duration")
            : specfem::io::sources_impl::get_real(fields, "hdurorf0");

    // Coordinates are resolved to global (x,y,z) at assembly time. Depth is
    // in km here.
    std::unique_ptr<specfem::coordinate_systems::coordinates<dim3>> coords;
    if (fields.contains("latitude") && fields.contains("longitude")) {
      double lon = specfem::io::sources_impl::get_double(fields, "longitude");
      double lat = specfem::io::sources_impl::get_double(fields, "latitude");
      double depth_m =
          specfem::io::sources_impl::get_double(fields, "depth") * 1000.0;
      coords =
          std::make_unique<specfem::coordinate_systems::geographic_coordinates>(
              lon, lat, depth_m);
    } else if (fields.contains("z")) {
      double x = specfem::io::sources_impl::get_double(fields, "x");
      double y = specfem::io::sources_impl::get_double(fields, "y");
      coords = std::make_unique<
          specfem::coordinate_systems::cartesian_coordinates<dim3>>(
          x, y, specfem::io::sources_impl::get_double(fields, "z"),
          std::array<double, 3>{ 0.0, 0.0, 0.0 });
    } else {
      // nullopt origin: depth-based, resolved against topography at assembly.
      double x = specfem::io::sources_impl::get_double(fields, "x");
      double y = specfem::io::sources_impl::get_double(fields, "y");
      double depth_m =
          specfem::io::sources_impl::get_double(fields, "depth") * 1000.0;
      coords = std::make_unique<
          specfem::coordinate_systems::cartesian_coordinates<dim3>>(
          x, y, -depth_m, std::nullopt);
    }

    // STF type, factor, and force direction
    auto stf_type =
        specfem::io::sources_impl::get_int(fields, "source time function");
    auto factor =
        specfem::io::sources_impl::get_real(fields, "factor force source");
    // Direction components in the local East-North-Up frame, matching the
    // SPECFEM3D/SPECFEM3D_GLOBE FORCESOLUTION convention. The direction is used
    // as given (not normalised) and scaled by the factor.
    auto comp_east = specfem::io::sources_impl::get_real(
        fields, "component dir vect source e");
    auto comp_north = specfem::io::sources_impl::get_real(
        fields, "component dir vect source n");
    auto comp_up = specfem::io::sources_impl::get_real(
        fields, "component dir vect source z_up");

    // Force in the local East-North-Up frame. Kept as-read; it is rotated into
    // the Cartesian mesh frame at assembly, once the source location is placed
    // (a globe rotates by the local spherical frame; a regional mesh takes the
    // components as the mesh axes).
    const specfem::sources::GeographicForce geographic{ factor * comp_east,
                                                        factor * comp_north,
                                                        factor * comp_up };

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
        std::move(coords), geographic, std::move(stf_ptr), wavefield_type));
  }

  return sources;
}
