#include "specfem/coordinate_systems/geographic.hpp"

#include "specfem/utilities/is_close.hpp"

#include <sstream>

bool specfem::coordinate_systems::geographic_coordinates::operator==(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &other) const {
  const auto *o =
      dynamic_cast<const specfem::coordinate_systems::geographic_coordinates *>(
          &other);
  if (!o)
    return false;
  return specfem::utilities::is_close(longitude, o->longitude) &&
         specfem::utilities::is_close(latitude, o->latitude) &&
         specfem::utilities::is_close(depth, o->depth);
}

std::string specfem::coordinate_systems::geographic_coordinates::print() const {
  std::ostringstream os;
  os << "geographic(lon=" << longitude << ", lat=" << latitude
     << ", depth=" << depth << ")";
  return os.str();
}
