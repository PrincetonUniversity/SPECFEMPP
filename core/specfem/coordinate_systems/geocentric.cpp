#include "specfem/coordinate_systems/geocentric.hpp"

#include "specfem/utilities/is_close.hpp"

#include <cmath>
#include <numbers>
#include <sstream>
#include <stdexcept>

specfem::coordinate_systems::geocentric_coordinates
specfem::coordinate_systems::geocentric_coordinates::from_cartesian(
    const double x, const double y, const double z) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
    throw std::invalid_argument("Geocentric conversion requires finite xyz");
  }
  const double horizontal = std::hypot(x, y);
  const double radius = std::hypot(horizontal, z);
  if (!std::isfinite(radius)) {
    throw std::invalid_argument("Geocentric radius exceeds double range");
  }
  const double theta = radius == 0.0 ? 0.0 : std::atan2(horizontal, z);
  double phi = horizontal == 0.0 ? 0.0 : std::atan2(y, x);
  constexpr double two_pi = 2.0 * std::numbers::pi;
  if (phi < 0.0) {
    phi += two_pi;
  }
  // Addition can round a tiny negative longitude to exactly 2*pi.
  if (phi == 0.0 || phi >= two_pi) {
    phi = 0.0;
  }
  return { radius, theta, phi };
}

std::array<double, 3>
specfem::coordinate_systems::geocentric_coordinates::to_cartesian() const {
  return { r * std::sin(theta) * std::cos(phi),
           r * std::sin(theta) * std::sin(phi), r * std::cos(theta) };
}

bool specfem::coordinate_systems::geocentric_coordinates::operator==(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &other) const {
  const auto *o =
      dynamic_cast<const specfem::coordinate_systems::geocentric_coordinates *>(
          &other);
  if (!o)
    return false;
  return specfem::utilities::is_close(r, o->r) &&
         specfem::utilities::is_close(theta, o->theta) &&
         specfem::utilities::is_close(phi, o->phi);
}

std::string specfem::coordinate_systems::geocentric_coordinates::print() const {
  std::ostringstream os;
  os << "Geocentric(r=" << r << ", theta=" << theta << ", phi=" << phi << ")";
  return os.str();
}
