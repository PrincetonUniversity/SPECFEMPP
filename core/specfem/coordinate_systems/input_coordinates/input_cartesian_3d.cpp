#include "specfem/coordinate_systems/input_coordinates/input_cartesian_3d.hpp"

#include "specfem/utilities/is_close.hpp"

#include <sstream>

bool specfem::coordinate_systems::input_cartesian_3d::operator==(
    const input_coordinates<specfem::element::dimension_tag::dim3> &other)
    const {
  const auto *o = dynamic_cast<const input_cartesian_3d *>(&other);
  if (!o)
    return false;
  return specfem::utilities::is_close(data.x, o->data.x) &&
         specfem::utilities::is_close(data.y, o->data.y) &&
         specfem::utilities::is_close(data.z, o->data.z);
}

std::string specfem::coordinate_systems::input_cartesian_3d::print() const {
  std::ostringstream os;
  os << "input_cartesian_3d(x=" << data.x << ", y=" << data.y
     << ", z=" << data.z << ")";
  return os.str();
}
