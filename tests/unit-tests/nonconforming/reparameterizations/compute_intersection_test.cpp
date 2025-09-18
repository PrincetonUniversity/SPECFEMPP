#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"

#include <stdexcept>

#include "algorithms/locate_point.hpp"
#include "specfem/assembly/nonconforming_interfaces/dim2/impl/compute_intersection.hpp"
#include <gtest/gtest.h>

TEST(impl__compute_intersection, KnotCorrectness) {
  const int ngnod = 4;
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg1("coorg1", ngnod);
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg2("coorg2", ngnod);
  // element 1 lies on [0,1] x [0,1]
  coorg1(0) = { 0, 0 };
  coorg1(1) = { 1, 0 };
  coorg1(2) = { 1, 1 };
  coorg1(3) = { 0, 1 };

  // match element 1 (right) to element 2 (left)
  coorg2(0).x = 1;
  coorg2(1).x = 2;
  coorg2(2).x = 2;
  coorg2(3).x = 1;

  const Kokkos::View<type_real *, Kokkos::HostSpace> mortar_quad("mortar_quad",
                                                                 3);
  mortar_quad(0) = -1;
  mortar_quad(1) = 0;
  mortar_quad(2) = 1;

  // different vertical offsets for element 2
  for (const auto [coord_lo, coord_hi] :
       std::vector<std::pair<type_real, type_real> >{
           { 0, 1 }, { -1, 0.5 }, { 0.5, 1 }, { 1.5, 2.5 }, { -100, -0.1 } }) {
    coorg2(0).z = coord_lo;
    coorg2(1).z = coord_lo;
    coorg2(2).z = coord_hi;
    coorg2(3).z = coord_hi;

    if (coord_lo > 1 || coord_hi < 0) {
      EXPECT_THROW(specfem::assembly::nonconforming_interfaces_impl::
                       compute_intersection(
                           coorg1, coorg2, specfem::mesh_entity::type::right,
                           specfem::mesh_entity::type::left, mortar_quad),
                   std::runtime_error)
          << "Global coordinate intervals:\n"
          << "   side 1: [0, 1]\n"
          << "   side 2: [" << coord_lo << ", " << coord_hi << "]\n"
          << "There should be no intersection, causing `compute_intersection` "
             "to throw an error, but none was thrown.";
      continue;
    }

    type_real eps = 1e-3;

    // compute mortar explicitly
    type_real global_lo = std::max(coord_lo, type_real(0));
    type_real global_hi = std::min(coord_hi, type_real(1));

    // coord_lo -> -1,  coord_hi -> 1
    type_real side1_lo = global_lo * 2 - 1;
    type_real side1_hi = global_hi * 2 - 1;
    type_real side2_lo = 2 * (global_lo - coord_lo) / (coord_hi - coord_lo) - 1;
    type_real side2_hi = 2 * (global_hi - coord_lo) / (coord_hi - coord_lo) - 1;

    auto out =
        specfem::assembly::nonconforming_interfaces_impl::compute_intersection(
            coorg1, coorg2, specfem::mesh_entity::type::right,
            specfem::mesh_entity::type::left, mortar_quad);

    // compare against ground truth
    std::vector<std::pair<type_real, type_real> > expectations{
      { side1_lo, side2_lo },
      { (side1_lo + side1_hi) / 2, (side2_lo + side2_hi) / 2 },
      { side1_hi, side2_hi }
    };
    std::vector<std::pair<bool, bool> > fails(3);
    bool failed = false;
    for (int i = 0; i < mortar_quad.size(); i++) {
      fails[i] = { std::abs(out[i].first - expectations[i].first) > eps,
                   std::abs(out[i].second - expectations[i].second) > eps };
      if (fails[i].first || fails[i].second) {
        failed = true;
      }
    }

    // error?
    if (failed) {
      std::ostringstream oss;
      oss << "Global coordinate intervals:\n"
          << "   side 1: [0, 1]\n"
          << "   side 2: [" << coord_lo << ", " << coord_hi << "]\n"
          << "   intersection: [" << global_lo << ", " << global_hi << "]\n"
          << "     side 1 coords: [" << side1_lo << ", " << side1_hi << "]\n"
          << "     side 2 coords: [" << side2_lo << ", " << side2_hi << "]\n\n";
      oss << "Intersection knots (side 1 coordinates):\n";
      for (int i = 0; i < mortar_quad.size(); i++) {
        type_real measured = out[i].first;
        type_real expect = expectations[i].first;
        if (fails[i].first) {
          oss << "[✘] " << measured << " (expected: " << expect
              << ", err:" << std::scientific << std::showpos
              << (measured - expect) << ")\n"
              << std::fixed;
        } else {
          oss << "[✔] " << measured << "\n";
        }
      }
      oss << "Intersection knots (side 2 coordinates):\n";
      for (int i = 0; i < mortar_quad.size(); i++) {
        type_real measured = out[i].second;
        type_real expect = expectations[i].second;
        if (fails[i].second) {
          oss << "[✘] " << measured << " (expected: " << expect
              << ", err:" << std::scientific << std::showpos
              << (measured - expect) << ")\n"
              << std::fixed;
        } else {
          oss << "[✔] " << measured << "\n";
        }
      }
      FAIL() << oss.str();
    }
  }
}
