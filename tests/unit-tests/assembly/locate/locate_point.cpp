#include "algorithms/locate_point.hpp"
#include "../test_fixture/test_fixture.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>

using specfem::point::global_coordinates;
using specfem::point::local_coordinates;

void test_locate_point(
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {

  constexpr auto dim = specfem::dimension::type::dim2;

  const type_real xmin = assembly.mesh.xmin;
  const type_real xmax = assembly.mesh.xmax;
  const type_real zmin = assembly.mesh.zmin;
  const type_real zmax = assembly.mesh.zmax;

  // Select 4 points between the min and max values
  const type_real x1 = xmin + 0.265 * (xmax - xmin);
  const type_real x2 = xmin + 0.55 * (xmax - xmin);
  const type_real x3 = xmin + 0.775 * (xmax - xmin);
  const type_real x4 = xmin + 0.9 * (xmax - xmin);
  const type_real z1 = zmin + 0.265 * (zmax - zmin);
  const type_real z2 = zmin + 0.55 * (zmax - zmin);
  const type_real z3 = zmin + 0.775 * (zmax - zmin);
  const type_real z4 = zmin + 0.9 * (zmax - zmin);

  // Create a list of points to test
  std::vector<global_coordinates<dim> > points = {
    { x1, z1 }, { x2, z2 }, { x3, z3 }, { x4, z4 }
  };

  // Loop over the points and check if they are located correctly
  for (const auto &point : points) {
    const auto lcoord = specfem::algorithms::locate_point(point, assembly.mesh);

    // Check if the local coordinates are within the expected range
    if (std::abs(lcoord.xi) > 1.0 + 1e-3 ||
        std::abs(lcoord.gamma) > 1.0 + 1e-3) {
      std::ostringstream message;
      message << "Local coordinates out of bounds: \n"
              << "\tOriginal point: (" << std::scientific << point.x << ", "
              << point.z << ")\n"
              << "\tLocal coordinates: (" << lcoord.xi << ", " << lcoord.gamma
              << ")\n";

      throw std::runtime_error(message.str());
    }

    // Check if we can locate the point back to global coordinates
    const auto gcoord =
        specfem::algorithms::locate_point(lcoord, assembly.mesh);

    // Check if the located point is close to the original point
    if ((specfem::point::distance(point, gcoord) >
         1e-6 * specfem::point::distance(point, { 0.0, 0.0 })) &&
        (specfem::point::distance(point, { 0.0, 0.0 }) > 1e-6)) {

      std::ostringstream message;
      message << "Failed to locate point back to global coordinates: \n"
              << "\tOriginal point: (" << std::scientific << point.x << ", "
              << point.z << ")\n"
              << "\tLocated point: (" << gcoord.x << ", " << gcoord.z << ")\n";

      throw std::runtime_error(message.str());
    }
  }
}

TEST_F(Assembly2D, LocatePoint) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    specfem::assembly::assembly<specfem::dimension::type::dim2> assembly =
        std::get<5>(parameters);

    try {
      test_locate_point(assembly);

      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test: " << Test.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
}
