#include "mesh/dim3/coordinates/coordinates.hpp"
#include <iostream>

std::string
specfem::mesh::coordinates<specfem::dimension::type::dim3>::print() const {

  std::ostringstream message;
  message << "Coordinates parameters:\n"
          << "------------------------------\n"
          << "Number of spectral elements: " << nspec << "\n"
          << "Number of global nodes:..... " << nglob << "\n"
          << "Number of GLLX:............. " << ngllx << "\n"
          << "Number of GLLY:............. " << nglly << "\n"
          << "Number of GLLZ:............. " << ngllz << "\n"
          << "------------------------------\n";

  return message.str();
}

std::string specfem::mesh::coordinates<specfem::dimension::type::dim3>::print(
    int iglob) const {
  std::ostringstream message;
  message << "Coordinates parameters for global node " << iglob << ":\n"
          << "------------------------------\n"
          << "x: " << x(iglob) << "\n"
          << "y: " << y(iglob) << "\n"
          << "z: " << z(iglob) << "\n"
          << "------------------------------\n";

  return message.str();
}
