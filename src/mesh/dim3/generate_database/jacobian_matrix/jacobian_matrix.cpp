#include "mesh/dim3/generate_database/jacobian_matrix/jacobian_matrix.hpp"
#include <iostream>
#include <sstream>

std::string
specfem::mesh::jacobian_matrix<specfem::dimension::type::dim3>::print() const {
  std::ostringstream message;
  message << "Partial parameters:\n"
          << "------------------------------\n"
          << "Number of spectral elements: " << nspec << "\n"
          << "Number of GLLX:............. " << ngllx << "\n"
          << "Number of GLLY:............. " << nglly << "\n"
          << "Number of GLLZ:............. " << ngllz << "\n"
          << "------------------------------\n";
  return message.str();
}

std::string
specfem::mesh::jacobian_matrix<specfem::dimension::type::dim3>::print(
    int ispec, int igllx, int iglly, int igllz) const {

  std::ostringstream message;
  message << "Partial parameters for spectral element " << ispec << ":\n"
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

  return message.str();
}

std::string
specfem::mesh::jacobian_matrix<specfem::dimension::type::dim3>::print(
    int ispec, const std::string partial_name) const {

  std::ostringstream message;

  // Create array pointer to coordinates.x y or z depending on the partial_name
  const Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                     Kokkos::DefaultHostExecutionSpace> &array = [&]() {
    if (partial_name == "xix") {
      return this->xix;
    } else if (partial_name == "xiy") {
      return this->xiy;
    } else if (partial_name == "xiz") {
      return this->xiz;
    } else if (partial_name == "etax") {
      return this->etax;
    } else if (partial_name == "etay") {
      return this->etay;
    } else if (partial_name == "etaz") {
      return this->etaz;
    } else if (partial_name == "gammax") {
      return this->gammax;
    } else if (partial_name == "gammay") {
      return this->gammay;
    } else if (partial_name == "gammaz") {
      return this->gammaz;
    } else if (partial_name == "jacobian") {
      return this->jacobian;
    } else {
      throw std::runtime_error(
          "Invalid partial_name. partial_name must be xix, xiy, xiz, etax, "
          "etay, etaz, gammax, gammay, gammaz, or jacobian\n");
    }
  }();

  message << "Mapping parameters for spectral element " << ispec << ":\n"
          << "--------------------------------------------------\n"
          << "\n"
          << " |---> igllx\n"
          << " |\n"
          << " V\n"
          << "iglly\n"
          << "\n"
          << partial_name << ":\n";

  for (int igllz = 0; igllz < ngllz; igllz++) {
    message << "igllz=" << igllz << ": ";

    for (int iglly = 0; iglly < nglly; iglly++) {
      if (iglly > 0) {
        message << "         ";
      }
      for (int igllx = 0; igllx < ngllx; igllx++) {
        message << array(ispec, igllx, iglly, igllz) << " ";
      }
      message << "\n";
    };
    message << "\n";
  };
  message << "------------------------------\n";

  return message.str();
}
