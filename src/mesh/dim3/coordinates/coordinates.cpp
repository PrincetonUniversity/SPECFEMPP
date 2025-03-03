#include "mesh/dim3/coordinates/coordinates.hpp"
#include "mesh/dim3/mapping/mapping.hpp"
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

std::string specfem::mesh::coordinates<specfem::dimension::type::dim3>::print(
    int ispec, specfem::mesh::mapping<specfem::dimension::type::dim3> &mapping,
    const std::string component) const {
  std::ostringstream message;
  int iglob;

  // Create array pointer to coordinates.x y or z depending on the component
  const Kokkos::View<type_real *, Kokkos::HostSpace> &array = [&]() {
    if (component == "x") {
      return this->x;
    } else if (component == "y") {
      return this->y;
    } else if (component == "z") {
      return this->z;
    } else {
      throw std::runtime_error(
          "Invalid component. component must be x, y, or z\n");
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
          << component << ":\n";

  for (int igllz = 0; igllz < ngllz; igllz++) {
    message << "igllz=" << igllz << ": ";

    for (int iglly = 0; iglly < nglly; iglly++) {
      if (iglly > 0) {
        message << "         ";
      }
      for (int igllx = 0; igllx < ngllx; igllx++) {
        iglob = mapping.ibool(ispec, igllx, iglly, igllz);
        message << array(iglob) << " ";
      }
      message << "\n";
    };
    message << "\n";
  };
  message << "------------------------------\n";

  return message.str();
}
