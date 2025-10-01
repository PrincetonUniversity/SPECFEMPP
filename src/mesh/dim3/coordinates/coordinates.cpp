#include "mesh/dim3/coordinates/coordinates.hpp"
#include "mesh/dim3/mapping/mapping.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <iostream>
#include <sstream>

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

/**
 * @brief Compute bounding box of the mesh`
 *
 */
std::array<type_real, 6>
specfem::mesh::coordinates<specfem::dimension::type::dim3>::bounding_box()
    const {
  std::array<type_real, 6> bbox;

  const int n = x.extent(0);

  type_real x_min, x_max, y_min, y_max, z_min, z_max;

  Kokkos::parallel_reduce(
      "specfem::mesh::coordinates::bounding_box",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
      [=](const int i, type_real &min_x, type_real &max_x, type_real &min_y,
          type_real &max_y, type_real &min_z, type_real &max_z) {
        if (x(i) < min_x)
          min_x = x(i);
        if (x(i) > max_x)
          max_x = x(i);
        if (y(i) < min_y)
          min_y = y(i);
        if (y(i) > max_y)
          max_y = y(i);
        if (z(i) < min_z)
          min_z = z(i);
        if (z(i) > max_z)
          max_z = z(i);
      },
      Kokkos::Min<type_real>(x_min), Kokkos::Max<type_real>(x_max),
      Kokkos::Min<type_real>(y_min), Kokkos::Max<type_real>(y_max),
      Kokkos::Min<type_real>(z_min), Kokkos::Max<type_real>(z_max));

  bbox[0] = x_min;
  bbox[1] = x_max;
  bbox[2] = y_min;
  bbox[3] = y_max;
  bbox[4] = z_min;
  bbox[5] = z_max;

  return bbox;
}
