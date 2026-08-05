#include "specfem/coordinate_systems/geocentric.hpp"

#include "specfem/utilities/is_close.hpp"

#include <sstream>

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
