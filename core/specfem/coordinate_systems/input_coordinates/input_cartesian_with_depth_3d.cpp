#include "specfem/coordinate_systems/input_coordinates/input_cartesian_with_depth_3d.hpp"

#include "specfem/utilities/is_close.hpp"

#include <sstream>

bool specfem::coordinate_systems::input_cartesian_with_depth_3d::operator==(
    const input_coordinates<specfem::element::dimension_tag::dim3> &other)
    const {
  const auto *o = dynamic_cast<const input_cartesian_with_depth_3d *>(&other);
  if (!o)
    return false;
  return specfem::utilities::is_close(x, o->x) &&
         specfem::utilities::is_close(y, o->y) &&
         specfem::utilities::is_close(depth, o->depth);
}

std::string
specfem::coordinate_systems::input_cartesian_with_depth_3d::print() const {
  std::ostringstream os;
  os << "input_cartesian_with_depth_3d(x=" << x << ", y=" << y
     << ", depth=" << depth << ")";
  return os.str();
}
