#include "specfem/coordinate_systems/coordinates/cartesian_3d.hpp"

#include <sstream>

namespace specfem {
namespace coordinate_systems {

std::string cartesian_3d::print() const {
  std::ostringstream os;
  os << "cartesian_3d(x=" << data.x << ", y=" << data.y << ", z=" << data.z
     << ")";
  return os.str();
}

} // namespace coordinate_systems
} // namespace specfem
