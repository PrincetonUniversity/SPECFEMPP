#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/datetime.hpp"
#include "specfem/io/sources/impl/reader.hpp"
#include "specfem/io/sources/impl/solution_format_helpers.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/units.hpp"

#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim3>>>
specfem::io::sources_impl::read<specfem::element::dimension_tag::dim3,
                                specfem::enums::source_format::CMTSOLUTION>(
    const std::string &file_path, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type) {

  constexpr int NLINES_PER_SOURCE = 13;
  constexpr auto dim3 = specfem::element::dimension_tag::dim3;

  auto lines = specfem::io::sources_impl::read_nonempty_lines(file_path);

  if (lines.empty() || lines.size() % NLINES_PER_SOURCE != 0) {
    throw std::runtime_error(
        "CMTSOLUTION file must contain a multiple of 13 non-empty lines, got " +
        std::to_string(lines.size()) + " in " + file_path);
  }

  const int nsources = static_cast<int>(lines.size()) / NLINES_PER_SOURCE;

  std::vector<std::shared_ptr<specfem::sources::source<dim3>>> sources;
  sources.reserve(nsources);

  for (int isrc = 0; isrc < nsources; ++isrc) {
    const int base = isrc * NLINES_PER_SOURCE;

    // Line 0: PDE header — parse datetime via tokenization
    int year, month, day, hour, minute;
    double second;
    {
      std::istringstream iss(lines[base]);
      std::string token;
      iss >> token; // skip "PDE" or similar label
      if (!(iss >> year >> month >> day >> hour >> minute >> second)) {
        throw std::runtime_error(
            "Failed to parse PDE header datetime in CMTSOLUTION: " +
            lines[base]);
      }
    }
    auto starttime =
        specfem::datetime::make(year, month, day, hour, minute, second);

    // Build field map from remaining 12 key:value lines
    auto fields = specfem::io::sources_impl::build_field_map(
        std::span<const std::string>(lines).subspan(base + 1, 12));

    // Time shift and half duration
    auto tshift = specfem::io::sources_impl::get_real(fields, "time shift");
    auto hdur = specfem::io::sources_impl::get_real(fields, "half duration");

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

    // Moment tensor — parse as DyneCentimeter, detect Cartesian vs geographic
    using namespace specfem::units::unit_symbols;
    using specfem::units::DyneCentimeter;
    using specfem::units::NewtonMeter;

    // Unit conversion: dyne-cm to N-m
    auto to_Nm = [](DyneCentimeter v) {
      return specfem::units::unit_cast<NewtonMeter>(v).raw();
    };

    auto stf_ptr = std::make_unique<specfem::source_time_functions::Heaviside>(
        nsteps, dt, hdur, tshift, 1.0, false);

    std::shared_ptr<specfem::sources::moment_tensor<dim3>> src;
    if (fields.contains("mrr")) {
      // Geographic (r, theta, phi) moment tensor. Kept as-read; it is rotated
      // into the Cartesian mesh frame at assembly, once the source location is
      // placed (a globe rotates by the local spherical frame, a regional mesh
      // applies the flat East-North-Up relabel).
      const specfem::sources::GeographicMomentTensor geographic{
        to_Nm(specfem::io::sources_impl::get_real(fields, "mrr") * dyn_cm),
        to_Nm(specfem::io::sources_impl::get_real(fields, "mtt") * dyn_cm),
        to_Nm(specfem::io::sources_impl::get_real(fields, "mpp") * dyn_cm),
        to_Nm(specfem::io::sources_impl::get_real(fields, "mrt") * dyn_cm),
        to_Nm(specfem::io::sources_impl::get_real(fields, "mrp") * dyn_cm),
        to_Nm(specfem::io::sources_impl::get_real(fields, "mtp") * dyn_cm)
      };
      src = std::make_shared<specfem::sources::moment_tensor<dim3>>(
          std::move(coords), geographic, std::move(stf_ptr), wavefield_type);
    } else {
      auto Mxx = specfem::io::sources_impl::get_real(fields, "mxx") * dyn_cm;
      auto Myy = specfem::io::sources_impl::get_real(fields, "myy") * dyn_cm;
      auto Mzz = specfem::io::sources_impl::get_real(fields, "mzz") * dyn_cm;
      auto Mxy = specfem::io::sources_impl::get_real(fields, "mxy") * dyn_cm;
      auto Mxz = specfem::io::sources_impl::get_real(fields, "mxz") * dyn_cm;
      auto Myz = specfem::io::sources_impl::get_real(fields, "myz") * dyn_cm;
      src = std::make_shared<specfem::sources::moment_tensor<dim3>>(
          std::move(coords), to_Nm(Mxx), to_Nm(Myy), to_Nm(Mzz), to_Nm(Mxy),
          to_Nm(Mxz), to_Nm(Myz), std::move(stf_ptr), wavefield_type);
    }
    src->set_starttime(starttime);
    sources.push_back(std::move(src));
  }

  return sources;
}
