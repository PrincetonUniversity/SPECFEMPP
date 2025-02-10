#include "mesh/modifiers/modifiers.hpp"
#include <cstdio>

// apply() handled in apply*.cpp in this directory

//===== display / debug / info =====
std::string specfem::mesh::modifiers::subdivisions_to_string() const {
  std::string repr =
      "subdivisions (set: " + std::to_string(subdivisions.size()) + "):";
#define BUFSIZE 50
  char buf[BUFSIZE];
  for (const auto &[matID, subs] : subdivisions) {
    std::snprintf(buf, BUFSIZE, "\n  - material %d: (nz,nx) = (%d,%d)", matID,
                  subs.first, subs.second);
    repr += std::string(buf, BUFSIZE);
  }
#undef BUFSIZE
  return repr;
}

std::string specfem::mesh::modifiers::to_string() const {
  std::string repr = "mesh modifiers: \n";
  repr += subdivisions_to_string();

  return repr;
}

//===== setting modifiers =====
void specfem::mesh::modifiers::set_subdivision(const int material, const int subdivide_z, const int subdivide_x){
  subdivisions.insert({material,std::make_pair(subdivide_z,subdivide_x)});
}
//===== getting modifiers =====
std::pair<int,int> specfem::mesh::modifiers::get_subdivision(const int material) const{
  auto got = subdivisions.find(material);
  if(got == subdivisions.end()){
    //default: no subdividing (1 subdiv in z, 1 in x)
    return std::make_pair(1,1);
  }else{
    return got -> second;
  }
}