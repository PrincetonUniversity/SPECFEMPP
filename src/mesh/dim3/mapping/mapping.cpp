
#include "mesh/dim3/mapping/mapping.hpp"
#include <iostream>

void specfem::mesh::mapping<specfem::dimension::type::dim3>::print() const {
  std::cout << "Mapping parameters:\n"
            << "------------------------------\n"
            << "Number of spectral elements: " << nspec << "\n"
            << "Number of global nodes: " << nglob << "\n"
            << "Number of irregular spectral elements: " << nspec_irregular
            << "\n"
            << "Number of GLLX: " << ngllx << "\n"
            << "Number of GLLY: " << nglly << "\n"
            << "Number of GLLZ: " << ngllz << "\n"
            << "------------------------------\n";
}

void specfem::mesh::mapping<specfem::dimension::type::dim3>::print(
    int ispec) const {
  std::cout << "Mapping parameters for spectral element " << ispec << ":\n"
            << "------------------------------\n"
            << "ibool:\n";
  for (int igllz = 0; igllz < ngllz; igllz++) {
    for (int iglly = 0; iglly < nglly; iglly++) {
      for (int igllx = 0; igllx < ngllx; igllx++) {
        std::cout << ibool(ispec, igllx, iglly, igllz) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "------------------------------\n";
}
