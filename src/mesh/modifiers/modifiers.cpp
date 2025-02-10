#include "mesh/modifiers/modifiers.hpp"
#include <cstdio>

std::string specfem::mesh::modifiers::subdivisions_to_string() {
  std::string repr =
      "subdivisions (set: " + std::to_string(subdivisions.size()) + "):";
#define BUFSIZE 50
  char buf[BUFSIZE];
  for (const auto &[matID, [ subz, subx ]] : subdivisions) {
    std::snprintf(buf, BUFSIZE, "\n  - material %d: (nz,nx) = (%d,%d)", matID,
                  subz, subx);
    repr += std::string(buf, BUFSIZE);
  }
#undef BUFSIZE
}

std::string specfem::mesh::modifiers::to_string() {
  std::string repr = "mesh modifiers: \n";
  repr += subdivisions_to_string();

  return repr;
}
