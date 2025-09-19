
#include "mesh/dim3/generate_database/mapping/mapping.hpp"
#include <iostream>
#include <sstream>

std::string
specfem::mesh::mapping<specfem::dimension::type::dim3>::print() const {

  std::ostringstream message;

  message << "Mapping parameters:\n"
          << "------------------------------\n"
          << "Number of spectral elements: " << nspec << "\n"
          << "Number of global nodes: " << nglob << "\n"
          << "Number of irregular spectral elements: " << nspec_irregular
          << "\n"
          << "Number of GLLX: " << ngllx << "\n"
          << "Number of GLLY: " << nglly << "\n"
          << "Number of GLLZ: " << ngllz << "\n"
          << "------------------------------\n";

  return message.str();
}

std::string
specfem::mesh::mapping<specfem::dimension::type::dim3>::print(int ispec) const {

  std::ostringstream message;

  message << "Mapping parameters for spectral element " << ispec << ":\n"
          << "--------------------------------------------------\n"
          << "\n"
          << " |---> igllx\n"
          << " |\n"
          << " V\n"
          << "iglly\n"
          << "\n"
          << "ibool:\n";
  for (int igllz = 0; igllz < ngllz; igllz++) {
    message << "igllz=" << igllz << ": ";
    for (int iglly = 0; iglly < nglly; iglly++) {
      if (iglly > 0) {
        message << "         ";
      }
      for (int igllx = 0; igllx < ngllx; igllx++) {
        message << ibool(ispec, igllx, iglly, igllz) << " ";
      }
      message << "\n";
    }
    message << "\n";
  }
  message << "------------------------------\n";

  return message.str();
}
