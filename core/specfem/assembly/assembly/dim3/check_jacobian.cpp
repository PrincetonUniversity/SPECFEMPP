#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/globe/radial_flags.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace specfem::assembly::check_jacobian_impl {

// Coordinates are in metres; latitude is geocentric, longitude is
// east-positive.
void append_globe_location(std::ostream &message, const type_real x,
                           const type_real y, const type_real z) {
  const type_real radius = std::hypot(x, y, z);
  const type_real horizontal_radius = std::hypot(x, y);
  constexpr type_real radians_to_degrees = 180 / std::numbers::pi_v<type_real>;
  message << " (m) radius_km=" << radius / 1000;
  if (std::isfinite(radius) && radius > 0) {
    message << " geocentric_lat_deg="
            << std::atan2(z, horizontal_radius) * radians_to_degrees;
  } else {
    message << " geocentric_lat_deg=undefined";
  }
  if (std::isfinite(radius) && horizontal_radius > 0) {
    message << " lon_deg=" << std::atan2(y, x) * radians_to_degrees;
  } else {
    message << " lon_deg=undefined";
  }
}

} // namespace specfem::assembly::check_jacobian_impl

void specfem::assembly::assembly<
    specfem::element::dimension_tag::dim3>::check_jacobian_matrix() const {
  const auto result = this->jacobian_matrix.check_small_jacobian();
  if (!result.found) {
    return;
  }

  const auto &diagnostics = result.diagnostics;
  constexpr std::size_t maximum_reported_elements = 10;
  std::ostringstream message;
  message << std::setprecision(std::numeric_limits<type_real>::max_digits10);
  message << "Invalid Jacobian mapping in " << diagnostics.size()
          << " element(s): determinants must be finite, strictly positive, and "
             "at least 1e-6 of the element-local mean absolute determinant.\n";
  message << "Invalid elements (compute order, up to 10):\n";

  const std::size_t report_count =
      std::min(maximum_reported_elements, diagnostics.size());
  for (std::size_t i = 0; i < report_count; ++i) {
    const auto &diagnostic = diagnostics[i];
    const int ispec = diagnostic.element_index;
    message << "  element=" << ispec
            << " mesh_element=" << this->mesh.h_compute_to_mesh(ispec)
            << " gll=(" << diagnostic.ix << ',' << diagnostic.iy << ','
            << diagnostic.iz << ") jacobian=" << diagnostic.jacobian
            << " scale=" << diagnostic.scale
            << " relative=" << diagnostic.relative_jacobian();
    if (!std::isfinite(diagnostic.jacobian)) {
      message << " reason=nonfinite_determinant";
    } else if (diagnostic.jacobian <= 0) {
      message << " reason=nonpositive_determinant";
    } else {
      message << " reason=small_relative_determinant";
    }
    const type_real x = this->mesh.h_coord(ispec, diagnostic.iz, diagnostic.iy,
                                           diagnostic.ix, 0);
    const type_real y = this->mesh.h_coord(ispec, diagnostic.iz, diagnostic.iy,
                                           diagnostic.ix, 1);
    const type_real z = this->mesh.h_coord(ispec, diagnostic.iz, diagnostic.iy,
                                           diagnostic.ix, 2);
    message << "\n    x=" << x << " y=" << y << " z=" << z;
    if (this->element_types.has_element_context()) {
      specfem::assembly::check_jacobian_impl::append_globe_location(message, x,
                                                                    y, z);
      const type_real rmin = this->element_types.rmin(ispec);
      const type_real rmax = this->element_types.rmax(ispec);
      message << "\n    region="
              << specfem::element::to_string(
                     this->element_types.get_region_tag(ispec))
              << " radial_range_km=[" << rmin / 1000 << ',' << rmax / 1000
              << ']' << " idoubling=" << this->element_types.idoubling(ispec);
      if (specfem::globe::is_central_cube(
              this->element_types.idoubling(ispec))) {
        message << " central_cube=true";
      }
    }
    message << '\n';
  }
  if (diagnostics.size() > report_count) {
    message << "  ... " << diagnostics.size() - report_count
            << " additional invalid element(s) omitted\n";
  }

  throw std::runtime_error(message.str());
}
