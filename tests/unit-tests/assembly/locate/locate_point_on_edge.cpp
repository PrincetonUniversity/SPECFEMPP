#include "../test_fixture/test_fixture.hpp"
#include "algorithms/locate_point.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem/point.hpp"
#include "specfem/point/coordinates.hpp"
#include <gtest/gtest.h>
#include <utility>

using specfem::point::global_coordinates;
using specfem::point::local_coordinates;

void test_locate_point_on_edge(
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {

  constexpr auto dim = specfem::dimension::type::dim2;

  const type_real xmin = assembly.mesh.xmin;
  const type_real xmax = assembly.mesh.xmax;
  const type_real zmin = assembly.mesh.zmin;
  const type_real zmax = assembly.mesh.zmax;

  const type_real eps_local = 1e-6;

  /* =============================
   * - test_locate_point_on_edge -
   * =============================
   *
   * For each trial, choose an element and a point on the
   * perimeter of that element. locate_point_on_edge should give the correct
   * value.
   */
  constexpr int num_trials = 10;

  const int nspec = assembly.get_total_number_of_elements();

  // sets xi,gamma constraint on edge, returns free (edge) coordinate.
  const auto get_edge_coordinate =
      [](type_real &xi, type_real &gamma,
         const specfem::mesh_entity::type &edgetype) -> type_real & {
    if (edgetype == specfem::mesh_entity::type::bottom ||
        edgetype == specfem::mesh_entity::type::top) {
      gamma = (edgetype == specfem::mesh_entity::type::bottom) ? -1 : 1;
      return xi;
    } else {
      xi = (edgetype == specfem::mesh_entity::type::left) ? -1 : 1;
      return gamma;
    }
  };

  // we could seed by anything. Here's one:
  std::srand(nspec);

  for (int itrial = 0; itrial < num_trials; itrial++) {
    const int ispec = std::rand() % nspec;
    const int edge_select = std::rand() % 4;
    specfem::mesh_entity::type edge;
    if (edge_select == 0) {
      edge = specfem::mesh_entity::type::bottom;
    } else if (edge_select == 1) {
      edge = specfem::mesh_entity::type::right;
    } else if (edge_select == 2) {
      edge = specfem::mesh_entity::type::top;
    } else {
      edge = specfem::mesh_entity::type::left;
    }

    type_real xi_target, gamma_target;

    type_real &local_target =
        get_edge_coordinate(xi_target, gamma_target, edge);
    local_target = ((((type_real)std::rand()) / RAND_MAX) * 2) - 1;

    // target point on perimeter selected. Get global coordinates.
    const auto global_coords = specfem::algorithms::locate_point(
        specfem::point::local_coordinates<specfem::dimension::type::dim2>(
            ispec, xi_target, gamma_target),
        assembly.mesh);

    // attempt to recover local edge coordinate
    const type_real local_test = specfem::algorithms::locate_point_on_edge(
        global_coords, assembly.mesh, ispec, edge);

    // Check if the local coordinates are within the expected range
    if (std::abs(local_test) > 1.0 + 1e-3) {
      std::ostringstream message;
      message << "Local coordinates out of bounds: \n"
              << "\tOriginal point: (" << std::scientific << global_coords.x
              << ", " << global_coords.z << ")\n"
              << "\tLocal coordinates: (" << xi_target << ", " << gamma_target
              << ")\n";

      throw std::runtime_error(message.str());
    }

    if (std::abs(local_test - local_target) > 1.0 + 1e-3) {
      std::ostringstream message;
      message << "Failed to locate point along edge: \n"
              << "\tOriginal point: " << local_target << "\n"
              << "\tLocated point:  " << local_test << "\n";

      throw std::runtime_error(message.str());
    }

    std::cout << "(" << xi_target << ", " << gamma_target << ") -- "
              << local_target << ": " << local_test << std::endl;
  }
}

TEST_F(ASSEMBLY, LocatePointOnEdge) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    specfem::assembly::assembly<specfem::dimension::type::dim2> assembly =
        std::get<5>(parameters);

    try {
      test_locate_point_on_edge(assembly);

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
