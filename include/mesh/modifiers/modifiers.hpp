#pragma once

#include "enumerations/dimension.hpp"
#include "mesh/mesh.hpp"

#include <string>
#include <unordered_map>

namespace specfem {
namespace mesh {

// dimtuple<T, dim> gives tuple<T,...> (size = dim)
// dimtuple<T, toset, set> gives tuple<T,...> (size = toset + sizeof(set))
template <typename T, int toset, typename... set> struct dimtuple {
  // using type = typename std::conditional<toset <= 0, std::tuple<set...>,
  // typename dimtuple<T,toset-1,T,set...>::type>::type;
  using unravelstruct = dimtuple<T, toset - 1, T, set...>;
  using type = typename unravelstruct::type;
  static inline std::string subdiv_str(type tup) {
    return unravelstruct::subdiv_str(tup) + ((toset > 1) ? "x" : "") +
           std::to_string(std::get<toset - 1>(tup));
  }
};
template <typename T, typename... set> struct dimtuple<T, 0, set...> {
  using type = std::tuple<set...>;
  static inline std::string subdiv_str(type tup) { return ""; }
};

template <specfem::dimension::type DimensionType> class modifiers {
public:
  static constexpr int dim = specfem::dimension::dimension<DimensionType>::dim;
  using subdiv_tuple = typename dimtuple<int, dim>::type;
  modifiers() {}

  //===== application =====
  void apply(specfem::mesh::mesh<DimensionType> &mesh) const;

  //===== display / debug / info =====
  std::string to_string() const;
  std::string subdivisions_to_string() const;

  //===== setting modifiers =====
  void set_subdivision(const int material, subdiv_tuple subdivisions);
  //===== getting modifiers =====
  specfem::mesh::modifiers<DimensionType>::subdiv_tuple
  get_subdivision(const int material) const;

private:
  std::unordered_map<int, subdiv_tuple> subdivisions; ///< map
                                                      ///< materialID ->
                                                      ///< (subdivide_z,
                                                      ///< subdivide_x)
};
} // namespace mesh
} // namespace specfem
