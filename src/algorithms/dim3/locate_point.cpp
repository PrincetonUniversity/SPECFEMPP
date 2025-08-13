#include "algorithms/locate_point.hpp"
#include "specfem/assembly.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"

namespace {

using MeshHostCoordinatesViewType =
    Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

std::tuple<int, int, int, int> rough_location(
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &global,
    const MeshHostCoordinatesViewType coord) {

  /***
   *  Roughly locate closest quadrature point to the source
   ***/

  const int nspec = coord.extent(0);
  const int ngllz = coord.extent(1);
  const int nglly = coord.extent(3);
  const int ngllx = coord.extent(4);

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
  }

  return std::make_tuple(xi, eta, gamma);
}

} // namespace

specfem::point::local_coordinates<specfem::dimension::type::dim3>
specfem::algorithms::locate_point(
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh) {

  const auto global_coordinates = mesh.h_coord;
  const auto index_mapping = mesh.h_index_mapping;
  const auto xi = mesh.h_xi;
  const auto eta = mesh.h_xi;
  const auto gamma = mesh.h_xi;
  const auto shape3D = mesh.h_shape3D;
  const int ngnod = mesh.ngnod;
  const int N = mesh.ngllx;

  int ix_guess, iy_guess, iz_guess, ispec_guess;

  std::tie(ispec_guess, ix_guess, iy_guess, iz_guess) =
      rough_location(coordinates, global_coordinates);

  const auto best_candidates = get_best_candidates(ispec_guess, index_mapping);

  type_real final_dist = std::numeric_limits<type_real>::max();

  int ispec_selected_source;
  type_real xi_selected, eta_selected, gamma_selected;

  const Kokkos::View<
      point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  for (auto &ispec : best_candidates) {
    auto sv_shape3D =
        Kokkos::subview(shape3D, iz_guess, iy_guess, ix_guess, Kokkos::ALL);

    type_real xi_guess = xi(ix_guess);
    type_real eta_guess = eta(iy_guess);
    type_real gamma_guess = gamma(iz_guess);

    for (int i = 0; i < ngnod; i++) {
      coorg(i).x = mesh.h_control_node_coordinates(ispec, i, 0);
      coorg(i).y = mesh.h_control_node_coordinates(ispec, i, 1);
      coorg(i).z = mesh.h_control_node_coordinates(ispec, i, 2);
    }

    // find the best location
    std::tie(xi_guess, eta_guess, gamma_guess) =
        get_best_location(coordinates, coorg, xi_guess, eta_guess, gamma_guess);

    // compute the distance
    auto [x, y, z] = jacobian::compute_locations(coorg, mesh.ngnod, xi_guess,
                                                 eta_guess, gamma_guess);
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

    ix_guess = int(N / 2.0);
    iz_guess = int(N / 2.0);
  }

  return { ispec_selected_source, xi_selected, eta_selected, gamma_selected };
}

specfem::point::global_coordinates<specfem::dimension::type::dim3>
specfem::algorithms::locate_point(
    const specfem::point::local_coordinates<specfem::dimension::type::dim3>
        &coordinate,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real eta = coordinate.eta;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  for (int i = 0; i < ngnod; i++) {
    coorg(i).x = mesh.h_control_node_coordinates(ispec, i, 0);
    coorg(i).y = mesh.h_control_node_coordinates(ispec, i, 1);
    coorg(i).z = mesh.h_control_node_coordinates(ispec, i, 2);
  }

  return jacobian::compute_locations(coorg, ngnod, xi, eta, gamma);
}

// Except for the tests this function is not used in the codebase.
specfem::point::global_coordinates<specfem::dimension::type::dim3>
specfem::algorithms::locate_point(
    const specfem::kokkos::HostTeam::member_type &team_member,
    const specfem::point::local_coordinates<specfem::dimension::type::dim3>
        &coordinate,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real eta = coordinate.eta;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, ngnod), [&](const int i) {
        coorg(i).x = mesh.h_control_node_coordinates(ispec, i, 0);
        coorg(i).y = mesh.h_control_node_coordinates(ispec, i, 1);
        coorg(i).z = mesh.h_control_node_coordinates(ispec, i, 2);
      });

  team_member.team_barrier();

  return jacobian::compute_locations(coorg, ngnod, xi, eta, gamma);
}
