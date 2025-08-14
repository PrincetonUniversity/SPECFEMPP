#include "algorithms/locate_point_impl.hpp"
#include "algorithms/locate_point.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <sstream>
#include <stdexcept>

namespace specfem {
namespace algorithms {
namespace locate_point_impl {

// Expose helper functions from locate_point.cpp for unit testing
std::tuple<int, int, int> rough_location(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &global,
    const specfem::kokkos::HostView4d<type_real> coord) {

  const int nspec = coord.extent(1);
  const int ngllz = coord.extent(2);
  const int ngllx = coord.extent(3);

  type_real dist_min = std::numeric_limits<type_real>::max();
  int ispec_selected, ix_selected, iz_selected;

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int j = 0; j < ngllz; j++) {
      for (int i = 0; i < ngllx; i++) {
        const specfem::point::global_coordinates<specfem::dimension::type::dim2>
            cart_coord = { coord(0, ispec, j, i), coord(1, ispec, j, i) };
        const type_real distance = specfem::point::distance(global, cart_coord);
        if (distance < dist_min) {
          ispec_selected = ispec;
          ix_selected = i;
          iz_selected = j;
          dist_min = distance;
        }
      }
    }
  }

  return std::make_tuple(ix_selected, iz_selected, ispec_selected);
}

std::vector<int> get_best_candidates(
    const int ispec_guess,
    const Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        index_mapping) {

  const int nspec = index_mapping.extent(0);
  const int ngllx = index_mapping.extent(1);
  const int ngllz = index_mapping.extent(2);

  std::vector<int> iglob_guess;
  iglob_guess.push_back(index_mapping(ispec_guess, 0, 0));
  iglob_guess.push_back(index_mapping(ispec_guess, ngllz - 1, 0));
  iglob_guess.push_back(index_mapping(ispec_guess, 0, ngllx - 1));
  iglob_guess.push_back(index_mapping(ispec_guess, ngllz - 1, ngllx - 1));

  std::vector<int> ispec_candidates;
  ispec_candidates.push_back(ispec_guess);

  for (int ispec = 0; ispec < nspec; ispec++) {
    if (ispec == ispec_guess)
      continue;

    // loop over only corners
    for (int j : { 0, ngllz - 1 }) {
      for (int i : { 0, ngllx - 1 }) {
        // check if this element is in contact with initial guess
        if (std::find(iglob_guess.begin(), iglob_guess.end(),
                      index_mapping(ispec, j, i)) != iglob_guess.end()) {
          // do not count the element twice
          if (ispec_candidates.size() > 0 &&
              ispec_candidates[ispec_candidates.size() - 1] != ispec)
            ispec_candidates.push_back(ispec);
        }
      }
    }
  }

  return ispec_candidates;
}

std::tuple<type_real, type_real> get_local_coordinates(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &global,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    type_real xi, type_real gamma) {

  const int ngnod = coorg.extent(0);

  for (int iter_loop = 0; iter_loop < 100; iter_loop++) {
    auto loc = specfem::jacobian::compute_locations(coorg, ngnod, xi, gamma);
    auto jacobian =
        specfem::jacobian::compute_jacobian(coorg, ngnod, xi, gamma);

    type_real dx = -(loc.x - global.x);
    type_real dz = -(loc.z - global.z);

    type_real dxi = jacobian.xix * dx + jacobian.xiz * dz;
    type_real dgamma = jacobian.gammax * dx + jacobian.gammaz * dz;

    xi += dxi;
    gamma += dgamma;

    if (xi > 1.01)
      xi = 1.01;
    if (xi < -1.01)
      xi = -1.01;
    if (gamma > 1.01)
      gamma = 1.01;
    if (gamma < -1.01)
      gamma = -1.01;

    // Check for convergence
    if (std::abs(dxi) < 1e-12 && std::abs(dgamma) < 1e-12)
      break;
  }

  return std::make_tuple(xi, gamma);
}

// Core locate_point logic extracted for testability
specfem::point::local_coordinates<specfem::dimension::type::dim2>
locate_point_core(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::kokkos::HostView4d<type_real> &global_coordinates,
    const Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &index_mapping,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coord,
    const int ngnod, const int ngllx) {

  int ix_guess, iz_guess, ispec_guess;

  std::tie(ix_guess, iz_guess, ispec_guess) =
      rough_location(coordinates, global_coordinates);

  const auto best_candidates = get_best_candidates(ispec_guess, index_mapping);

  type_real final_dist = std::numeric_limits<type_real>::max();

  int ispec_selected = -1;
  type_real xi_selected = -9999.0;
  type_real gamma_selected = -9999.0;
  specfem::point::global_coordinates<specfem::dimension::type::dim2>
      coord_point;

  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  for (auto &ispec : best_candidates) {
    type_real xi_guess = 0.0; // Start at element center
    type_real gamma_guess = 0.0;

    // Extract control node coordinates for this element
    for (int i = 0; i < ngnod; i++) {
      coorg(i).x = control_node_coord(0, ispec, i);
      coorg(i).z = control_node_coord(1, ispec, i);
    }

    // Find the best location using Newton-Raphson
    std::tie(xi_guess, gamma_guess) =
        get_local_coordinates(coordinates, coorg, xi_guess, gamma_guess);

    // Compute the global coordinates from the found local coordinates
    auto coord_computed = specfem::jacobian::compute_locations(
        coorg, ngnod, xi_guess, gamma_guess);

    // Compute the distance from target to found location
    type_real dist = specfem::point::distance(coordinates, coord_computed);

    // Keep the best result
    if (dist < final_dist) {
      ispec_selected = ispec;
      xi_selected = xi_guess;
      gamma_selected = gamma_guess;
      coord_point = coord_computed;
      final_dist = dist;
    }
  }

  // Check if the found coordinates are valid
  bool xi_out_of_bounds = std::fabs(std::fabs(xi_selected) - 1.01) < 1e-6;
  bool gamma_out_of_bounds = std::fabs(std::fabs(gamma_selected) - 1.01) < 1e-6;
  bool ispec_invalid = ispec_selected < 0;

  // If the found coordinates are out of bounds, throw an error
  if (xi_out_of_bounds || gamma_out_of_bounds || ispec_invalid) {
    std::ostringstream oss;
    oss << "\nFailed to locate point in the mesh:\n"
        << "  (ispec, xi, gamma)   = (" << ispec_selected << ", " << xi_selected
        << ", " << gamma_selected << ")\n"
        << "  (target_x, target_z) = (" << coordinates.x << ", "
        << coordinates.z << ")\n"
        << "   (found_x,  found_z) = (" << coord_point.x << ", "
        << coord_point.z << ")\n"
        << "            final_dist = " << final_dist << "\n";
    throw std::runtime_error(oss.str());
  }

  return { ispec_selected, xi_selected, gamma_selected };
}

} // namespace locate_point_impl
} // namespace algorithms
} // namespace specfem
