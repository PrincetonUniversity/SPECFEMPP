#include "specfem/coordinate_systems/input_coordinates/input_geographic_3d.hpp"

#include "specfem/utilities/is_close.hpp"

#include <sstream>

bool specfem::coordinate_systems::input_geographic_3d::operator==(
    const input_coordinates<specfem::element::dimension_tag::dim3> &other)
    const {
  const auto *o = dynamic_cast<const input_geographic_3d *>(&other);
  if (!o)
    return false;
  return specfem::utilities::is_close(data.longitude, o->data.longitude) &&
         specfem::utilities::is_close(data.latitude, o->data.latitude) &&
         specfem::utilities::is_close(data.depth, o->data.depth);
}

std::string specfem::coordinate_systems::input_geographic_3d::print() const {
  std::ostringstream os;
  os << "input_geographic_3d(lon=" << data.longitude
     << ", lat=" << data.latitude << ", depth=" << data.depth << ")";
  return os.str();
}
