#include "specfem/coordinate_systems/coordinates/cartesian_3d.hpp"
#include "specfem/coordinate_systems/coordinates/cartesian_with_depth_3d.hpp"
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

    // Coordinates (x, y required; z or depth required)
    auto x = specfem::io::sources_impl::get_real(fields, "x");
    auto y = specfem::io::sources_impl::get_real(fields, "y");

    // Build generic coordinates — resolution to global (x,y,z) is deferred
    // to assembly time when topography/ellipticity are available.
    std::unique_ptr<specfem::coordinate_systems::coordinates<dim3>> coords;
    if (fields.contains("z")) {
      coords = std::make_unique<specfem::coordinate_systems::cartesian_3d>(
          x, y, specfem::io::sources_impl::get_real(fields, "z"));
    } else {
      // CMTSOLUTION depth is in km — convert to meters
      coords = std::make_unique<
          specfem::coordinate_systems::cartesian_with_depth_3d>(
          x, y, specfem::io::sources_impl::get_real(fields, "depth") * 1000.0);
    }

    // Moment tensor — parse as DyneCentimeter, detect Cartesian vs spherical
    using namespace specfem::units::unit_symbols;
    using specfem::units::DyneCentimeter;
    using specfem::units::NewtonMeter;

    DyneCentimeter Mxx(0), Myy(0), Mzz(0), Mxy(0), Mxz(0), Myz(0);

    if (fields.contains("mrr")) {
      // Spherical labels: convert Mrr/Mtt/Mpp -> Mxx/Myy/Mzz
      auto Mrr = specfem::io::sources_impl::get_real(fields, "mrr") * dyn_cm;
      auto Mtt = specfem::io::sources_impl::get_real(fields, "mtt") * dyn_cm;
      auto Mpp = specfem::io::sources_impl::get_real(fields, "mpp") * dyn_cm;
      auto Mrt = specfem::io::sources_impl::get_real(fields, "mrt") * dyn_cm;
      auto Mrp = specfem::io::sources_impl::get_real(fields, "mrp") * dyn_cm;
      auto Mtp = specfem::io::sources_impl::get_real(fields, "mtp") * dyn_cm;

      Mxx = Mpp;
      Myy = Mtt;
      Mzz = Mrr;
      Mxy = -Mtp;
      Mxz = Mrp;
      Myz = -Mrt;
    } else {
      Mxx = specfem::io::sources_impl::get_real(fields, "mxx") * dyn_cm;
      Myy = specfem::io::sources_impl::get_real(fields, "myy") * dyn_cm;
      Mzz = specfem::io::sources_impl::get_real(fields, "mzz") * dyn_cm;
      Mxy = specfem::io::sources_impl::get_real(fields, "mxy") * dyn_cm;
      Mxz = specfem::io::sources_impl::get_real(fields, "mxz") * dyn_cm;
      Myz = specfem::io::sources_impl::get_real(fields, "myz") * dyn_cm;
    }

    // Unit conversion: dyne-cm to N-m
    auto to_Nm = [](DyneCentimeter v) {
      return specfem::units::unit_cast<NewtonMeter>(v).raw();
    };

    // Construct STF and source
    auto stf_ptr =
        std::make_unique<specfem::source_time_functions::GaussianHdur>(
            nsteps, dt, hdur, tshift, 1.0, false);

    auto src = std::make_shared<specfem::sources::moment_tensor<dim3>>(
        std::move(coords), to_Nm(Mxx), to_Nm(Myy), to_Nm(Mzz), to_Nm(Mxy),
        to_Nm(Mxz), to_Nm(Myz), std::move(stf_ptr), wavefield_type);
    src->set_starttime(starttime);
    sources.push_back(std::move(src));
  }

  return sources;
}
