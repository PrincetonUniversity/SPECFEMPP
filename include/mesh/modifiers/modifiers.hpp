#pragma once

#include "mesh/mesh.hpp"


#include <utility>
#include <string>
#include <unordered_map>

namespace specfem {
namespace mesh {
class modifiers {
public:
  modifiers() {}
  
  //===== application =====
  template<specfem::dimension::type DimensionType>
  void apply(specfem::mesh::mesh<DimensionType>& mesh) const;

  //===== display / debug / info =====
  std::string to_string() const;
  std::string subdivisions_to_string() const;


  //===== setting modifiers =====
  void set_subdivision(const int material, const int subdivide_z, const int subdivide_x);
  //===== getting modifiers =====
  std::pair<int,int> get_subdivision(const int material) const;
private:
  std::unordered_map<int, std::pair<int, int> > subdivisions; ///< map
                                                              ///< materialID ->
                                                              ///< (subdivide_z,
                                                              ///< subdivide_x)
};
} // namespace mesh
} // namespace specfem
