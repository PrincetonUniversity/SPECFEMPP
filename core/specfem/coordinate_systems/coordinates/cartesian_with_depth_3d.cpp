#include "specfem/coordinate_systems/coordinates/cartesian_with_depth_3d.hpp"

#include <sstream>

namespace specfem {
namespace coordinate_systems {

std::string cartesian_with_depth_3d::print() const {
  std::ostringstream os;
  os << "cartesian_with_depth_3d(x=" << x << ", y=" << y << ", depth=" << depth
     << ")";
  return os.str();
}

} // namespace coordinate_systems
} // namespace specfem
