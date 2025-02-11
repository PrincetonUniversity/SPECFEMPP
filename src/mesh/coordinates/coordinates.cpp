#include "mesh/coordinates/coordinates.hpp"
#include <iostream>

void specfem::mesh::coordinates<specfem::dimension::type::dim3>::print() const {
  std::cout << "Coordinates parameters:\n"
            << "------------------------------\n"
            << "Number of spectral elements: " << nspec << "\n"
            << "Number of global nodes:..... " << nglob << "\n"
            << "Number of GLLX:............. " << ngllx << "\n"
            << "Number of GLLY:............. " << nglly << "\n"
            << "Number of GLLZ:............. " << ngllz << "\n"
            << "------------------------------\n";
}

void specfem::mesh::coordinates<specfem::dimension::type::dim3>::print(
    int iglob) const {
  std::cout << "Coordinates parameters for global node " << iglob << ":\n"
            << "------------------------------\n"
            << "x: " << x(iglob) << "\n"
            << "y: " << y(iglob) << "\n"
            << "z: " << z(iglob) << "\n"
            << "------------------------------\n";
}

void specfem::mesh::coordinates<specfem::dimension::type::dim3>::print(
    int ispec, int igllx, int iglly, int igllz) const {
  std::cout << "Coordinates parameters for spectral element " << ispec << ":\n"
            << "------------------------------\n"
            << "xix:..... " << xix(ispec, igllx, iglly, igllz) << "\n"
            << "xiy:..... " << xiy(ispec, igllx, iglly, igllz) << "\n"
            << "xiz:..... " << xiz(ispec, igllx, iglly, igllz) << "\n"
            << "etax:.... " << etax(ispec, igllx, iglly, igllz) << "\n"
            << "etay:.... " << etay(ispec, igllx, iglly, igllz) << "\n"
            << "etaz:.... " << etaz(ispec, igllx, iglly, igllz) << "\n"
            << "gammax:.. " << gammax(ispec, igllx, iglly, igllz) << "\n"
            << "gammay:.. " << gammay(ispec, igllx, iglly, igllz) << "\n"
            << "gammaz:.. " << gammaz(ispec, igllx, iglly, igllz) << "\n"
            << "jacobian: " << jacobian(ispec, igllx, iglly, igllz) << "\n"
            << "------------------------------\n";
}
