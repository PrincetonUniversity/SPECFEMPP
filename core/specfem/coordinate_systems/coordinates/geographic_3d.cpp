#include "specfem/coordinate_systems/coordinates/geographic_3d.hpp"

#include <sstream>

namespace specfem {
namespace coordinate_systems {

std::string geographic_3d::print() const {
  std::ostringstream os;
  os << "geographic_3d(lon=" << data.longitude << ", lat=" << data.latitude
     << ", depth=" << data.depth << ")";
  return os.str();
}

} // namespace coordinate_systems
} // namespace specfem
