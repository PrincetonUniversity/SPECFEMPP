#include "algorithms/locate_point_impl.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {
namespace locate_point_impl {

// 3D implementations moved from anonymous namespace

std::tuple<int, int, int, int> rough_location(
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &global,
    const MeshHostCoordinatesViewType3D coord) {

  const int nspec = coord.extent(0);
  const int ngllz = coord.extent(1);
  const int nglly = coord.extent(2);
  const int ngllx = coord.extent(3);

  type_real dist_min = std::numeric_limits<type_real>::max();
  int ispec_selected, ix_selected, iy_selected, iz_selected;

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int k = 0; k < ngllz; k++) {
      for (int j = 0; j < nglly; j++) {
        for (int i = 0; i < ngllx; i++) {

          // Get the global coordinates of the quadrature point
          const specfem::point::global_coordinates<
              specfem::dimension::type::dim3>
              cart_coord = { coord(ispec, k, j, i, 0), coord(ispec, k, j, i, 1),
                             coord(ispec, k, j, i, 2) };

          // Compute the distance between the global coordinates and the
          // quadrature point
          const type_real distance =
              specfem::point::distance(global, cart_coord);

          // If the distance is smaller than the minimum distance found so far,
          // update the selected quadrature point
          if (distance < dist_min) {
            ispec_selected = ispec;
            ix_selected = i;
            iy_selected = j;
            iz_selected = k;
            dist_min = distance;
          }
        }
      }
    }
  }

  return std::make_tuple(ispec_selected, ix_selected, iy_selected, iz_selected);
}

std::vector<int> get_best_candidates(
    const int ispec_guess,
    const Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        index_mapping) {

  const int nspec = index_mapping.extent(0);
  const int ngllz = index_mapping.extent(1);
  const int nglly = index_mapping.extent(2);
  const int ngllx = index_mapping.extent(3);

  std::vector<int> iglob_guess;
  // corners at gllz = 0
  iglob_guess.push_back(index_mapping(ispec_guess, 0, 0, 0));
  iglob_guess.push_back(index_mapping(ispec_guess, 0, 0, ngllx - 1));
  iglob_guess.push_back(index_mapping(ispec_guess, 0, nglly - 1, 0));
  iglob_guess.push_back(index_mapping(ispec_guess, 0, nglly - 1, ngllx - 1));
  // corners at gllz = ngllz - 1
  iglob_guess.push_back(index_mapping(ispec_guess, ngllz - 1, 0, 0));
  iglob_guess.push_back(index_mapping(ispec_guess, ngllz - 1, nglly - 1, 0));
  iglob_guess.push_back(index_mapping(ispec_guess, ngllz - 1, 0, ngllx - 1));
  iglob_guess.push_back(
      index_mapping(ispec_guess, ngllz - 1, nglly - 1, ngllx - 1));

  std::vector<int> ispec_candidates;
  ispec_candidates.push_back(ispec_guess);

  for (int ispec = 0; ispec < nspec; ispec++) {
    if (ispec == ispec_guess)
      continue;

    // loop over only corners
    for (int k : { 0, ngllz - 1 }) {
      for (int j : { 0, nglly - 1 }) {
        for (int i : { 0, ngllx - 1 }) {
          // check if this element is in contact with initial guess
          if (std::find(iglob_guess.begin(), iglob_guess.end(),
                        index_mapping(ispec, k, j, i)) != iglob_guess.end()) {
            // do not count the element twice
            if (ispec_candidates.size() > 0 &&
                ispec_candidates[ispec_candidates.size() - 1] != ispec)
              ispec_candidates.push_back(ispec);
          }
        }
      }
    }
  }

  return ispec_candidates;
}

std::tuple<type_real, type_real, type_real> get_best_location(
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &global,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    type_real xi, type_real eta, type_real gamma) {

  const int ngnod = coorg.extent(0);

  for (int iter_loop = 0; iter_loop < 100; iter_loop++) {
    auto loc =
        specfem::jacobian::compute_locations(coorg, ngnod, xi, eta, gamma);
    auto jacobian =
        specfem::jacobian::compute_jacobian(coorg, ngnod, xi, eta, gamma);

    // Compute the correction to the local coordinates
    type_real dx = -(loc.x - global.x);
    type_real dy = -(loc.y - global.y);
    type_real dz = -(loc.z - global.z);

    // Compute the change in local coordinates using the Jacobian
    type_real dxi = jacobian.xix * dx + jacobian.xiy * dy + jacobian.xiz * dz;
    type_real deta =
        jacobian.etax * dx + jacobian.etay * dy + jacobian.etaz * dz;
    type_real dgamma =
        jacobian.gammax * dx + jacobian.gammay * dy + jacobian.gammaz * dz;

    // Update the local coordinates
    xi += dxi;
    eta += deta;
    gamma += dgamma;

    // Clip the local coordinates to the (somewhat) valid range
    if (xi > 1.01)
      xi = 1.01;
    if (xi < -1.01)
      xi = -1.01;
    if (eta > 1.01)
      eta = 1.01;
    if (eta < -1.01)
      eta = -1.01;
    if (gamma > 1.01)
      gamma = 1.01;
    if (gamma < -1.01)
      gamma = -1.01;

    // Check convergence
    if (std::abs(dxi) < 1e-12 && std::abs(deta) < 1e-12 &&
        std::abs(dgamma) < 1e-12)
      break;
  }

  return std::make_tuple(xi, eta, gamma);
}

// Core locate_point logic extracted for testability
specfem::point::local_coordinates<specfem::dimension::type::dim3>
locate_point_core(
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &coordinates,
    const MeshHostCoordinatesViewType3D &global_coordinates,
    const Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &index_mapping,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coordinates,
    const int ngnod, const int ngllx) {

  int ix_guess, iy_guess, iz_guess, ispec_guess;

  std::tie(ispec_guess, ix_guess, iy_guess, iz_guess) =
      rough_location(coordinates, global_coordinates);

  const auto best_candidates = get_best_candidates(ispec_guess, index_mapping);

  type_real final_dist = std::numeric_limits<type_real>::max();

  int ispec_selected_source;
  type_real xi_selected, eta_selected, gamma_selected;

  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  for (auto &ispec : best_candidates) {
    type_real xi_guess = 0.0; // Start at element center
    type_real eta_guess = 0.0;
    type_real gamma_guess = 0.0;

    // Extract control node coordinates for this element
    for (int i = 0; i < ngnod; i++) {
      coorg(i).x = control_node_coordinates(ispec, i, 0);
      coorg(i).y = control_node_coordinates(ispec, i, 1);
      coorg(i).z = control_node_coordinates(ispec, i, 2);
    }

    // Find the best location using Newton-Raphson
    std::tie(xi_guess, eta_guess, gamma_guess) =
        get_best_location(coordinates, coorg, xi_guess, eta_guess, gamma_guess);

    // Compute the distance from target to found location
    auto [x, y, z] = specfem::jacobian::compute_locations(
        coorg, ngnod, xi_guess, eta_guess, gamma_guess);
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        cart_coord = { x, y, z };

    type_real dist = specfem::point::distance(coordinates, cart_coord);

    if (dist < final_dist) {
      ispec_selected_source = ispec;
      xi_selected = xi_guess;
      eta_selected = eta_guess;
      gamma_selected = gamma_guess;
      final_dist = dist;
    }
  }

  return { ispec_selected_source, xi_selected, eta_selected, gamma_selected };
}

} // namespace locate_point_impl
} // namespace algorithms
} // namespace specfem
