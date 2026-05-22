#include "specfem/coordinate_systems/input_coordinates/input_cartesian_2d.hpp"

#include "specfem/utilities/is_close.hpp"

#include <sstream>

bool specfem::coordinate_systems::input_cartesian_2d::operator==(
    const input_coordinates<specfem::element::dimension_tag::dim2> &other)
    const {
  const auto *o = dynamic_cast<const input_cartesian_2d *>(&other);
  if (!o)
    return false;
  return specfem::utilities::is_close(x, o->x) &&
         specfem::utilities::is_close(z, o->z);
}

std::string specfem::coordinate_systems::input_cartesian_2d::print() const {
  std::ostringstream os;
  os << "input_cartesian_2d(x=" << x << ", z=" << z << ")";
  return os.str();
}
