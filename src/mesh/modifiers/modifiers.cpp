#include "mesh/modifiers/modifiers.hpp"
#include <cstdio>
#include <string>

// apply() handled in apply*.cpp in this directory

//===== display / debug / info =====
template <specfem::dimension::type DimensionType>
std::string
specfem::mesh::modifiers<DimensionType>::subdivisions_to_string() const {
  std::string repr =
      "subdivisions (set: " + std::to_string(subdivisions.size()) + "):";
#define BUFSIZE 50
  char buf[BUFSIZE];
  for (const auto &[matID, subs] : subdivisions) {
    repr += "\n  - material %d: " + dimtuple<int, dim>::subdiv_str(subs) +
            " cell subdivision";
  }
#undef BUFSIZE
  return repr;
}

template <specfem::dimension::type DimensionType>
std::string specfem::mesh::modifiers<DimensionType>::to_string() const {
  std::string repr = "mesh modifiers: \n";
  repr += subdivisions_to_string();

  return repr;
}

//===== setting modifiers =====
template <specfem::dimension::type DimensionType>
void specfem::mesh::modifiers<DimensionType>::set_subdivision(
    const int material, subdiv_tuple subdivs) {
  subdivisions.insert(std::make_pair(0, subdivs));
}
//===== getting modifiers =====
template <specfem::dimension::type DimensionType>
typename specfem::mesh::modifiers<DimensionType>::subdiv_tuple
specfem::mesh::modifiers<DimensionType>::get_subdivision(
    const int material) const {
  auto got = subdivisions.find(material);
  if (got == subdivisions.end()) {
    // default: no subdividing (1 subdiv in z, 1 in x)
    return {};
  } else {
    return got->second;
  }
}

template class specfem::mesh::modifiers<specfem::dimension::type::dim2>;
template class specfem::mesh::modifiers<specfem::dimension::type::dim3>;
