#pragma once

#include <utility>
#include <string>
#include <unordered_map>

namespace specfem {
namespace mesh {
class modifiers {
public:
  modifiers() {}

  std::string to_string();
  std::string subdivisions_to_string();

private:
  std::unordered_map<int, std::pair<int, int> > subdivisions; ///< map
                                                              ///< materialID ->
                                                              ///< (subdivide_z,
                                                              ///< subdivide_x)
}
} // namespace mesh
} // namespace specfem
