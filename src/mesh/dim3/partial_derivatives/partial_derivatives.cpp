#include "mesh/dim3/partial_derivatives/partial_derivatives.hpp"
#include <iostream>

void specfem::mesh::partial_derivatives<specfem::dimension::type::dim3>::print()
    const {
  std::cout << "Partial parameters:\n"
            << "------------------------------\n"
            << "Number of spectral elements: " << nspec << "\n"
            << "Number of GLLX:............. " << ngllx << "\n"
            << "Number of GLLY:............. " << nglly << "\n"
            << "Number of GLLZ:............. " << ngllz << "\n"
            << "------------------------------\n";
}

void specfem::mesh::partial_derivatives<specfem::dimension::type::dim3>::print(
    int ispec, int igllx, int iglly, int igllz) const {
  std::cout << "Partial parameters for spectral element " << ispec << ":\n"
            << "------------------------------\n"
            << "dxid (x,y,z): " << xix(ispec, igllx, iglly, igllz) << "\n"
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
