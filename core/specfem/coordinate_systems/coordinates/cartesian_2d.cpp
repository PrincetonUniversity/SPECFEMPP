#include "specfem/coordinate_systems/coordinates/cartesian_2d.hpp"

#include <sstream>

namespace specfem {
namespace coordinate_systems {

std::string cartesian_2d::print() const {
  std::ostringstream os;
  os << "cartesian_2d(x=" << x << ", z=" << z << ")";
  return os.str();
}

} // namespace coordinate_systems
} // namespace specfem
