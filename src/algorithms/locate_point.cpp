#include "algorithms/locate_point.hpp"
#include "compute/compute_mesh.hpp"
#include "jacobian/interface.hpp"
#include "point/coordinates.hpp"

namespace {

std::tuple<int, int, int> rough_location(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &global,
    const specfem::kokkos::HostView4d<type_real> coord) {

  /***
   *  Roughly locate closest quadrature point to the source
   ***/

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

std::tuple<type_real, type_real> get_best_location(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &global,
    const specfem::kokkos::HostView2d<type_real> s_coord, type_real xi,
    type_real gamma) {

  const int ngnod = s_coord.extent(1);

  for (int iter_loop = 0; iter_loop < 100; iter_loop++) {
    auto [x, z] =
        specfem::jacobian::compute_locations(s_coord, ngnod, xi, gamma);
    auto [xix, xiz, gammax, gammaz] =
        specfem::jacobian::compute_inverted_derivatives(s_coord, ngnod, xi,
                                                        gamma);

    type_real dx = -(x - global.x);
    type_real dz = -(z - global.z);

    type_real dxi = xix * dx + xiz * dz;
    type_real dgamma = gammax * dx + gammaz * dz;

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
  }

  return std::make_tuple(xi, gamma);
}

} // namespace

specfem::point::local_coordinates<specfem::dimension::type::dim2>
specfem::algorithms::locate_point(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::compute::mesh &mesh) {

  const auto global_coordinates = mesh.points.h_coord;
  const auto index_mapping = mesh.points.h_index_mapping;
  const auto xi = mesh.quadratures.gll.h_xi;
  const auto gamma = mesh.quadratures.gll.h_xi;
  const auto shape2D = mesh.quadratures.gll.shape_functions.h_shape2D;
  const int ngnod = mesh.control_nodes.ngnod;
  const int N = mesh.quadratures.gll.N;

  int ix_guess, iz_guess, ispec_guess;

  std::tie(ix_guess, iz_guess, ispec_guess) =
      rough_location(coordinates, global_coordinates);

  const auto best_candidates = get_best_candidates(ispec_guess, index_mapping);

  type_real final_dist = std::numeric_limits<type_real>::max();

  int ispec_selected_source;
  type_real xi_source, gamma_source;

  specfem::kokkos::HostView2d<type_real> s_coord("s_coord", 2, ngnod);

  for (auto &ispec : best_candidates) {
    auto sv_shape2D = Kokkos::subview(shape2D, iz_guess, ix_guess, Kokkos::ALL);

    type_real xi_guess = xi(ix_guess);
    type_real gamma_guess = gamma(iz_guess);

    for (int i = 0; i < ngnod; i++) {
      s_coord(0, i) = mesh.control_nodes.h_coord(0, ispec, i);
      s_coord(1, i) = mesh.control_nodes.h_coord(1, ispec, i);
    }

    // find the best location
    std::tie(xi_guess, gamma_guess) =
        get_best_location(coordinates, s_coord, xi_guess, gamma_guess);

    // compute the distance
    auto [x, z] = jacobian::compute_locations(s_coord, mesh.control_nodes.ngnod,
                                              xi_guess, gamma_guess);
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        cart_coord = { x, z };

    type_real dist = specfem::point::distance(coordinates, cart_coord);

    if (dist < final_dist) {
      ispec_selected_source = ispec;
      xi_source = xi_guess;
      gamma_source = gamma_guess;
      final_dist = dist;
    }

    ix_guess = int(N / 2.0);
    iz_guess = int(N / 2.0);
  }

  return { ispec_selected_source, xi_source, gamma_source };
}

specfem::point::global_coordinates<specfem::dimension::type::dim2>
specfem::algorithms::locate_point(
    const specfem::point::local_coordinates<specfem::dimension::type::dim2>
        &coordinate,
    const specfem::compute::mesh &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.control_nodes.ngnod;

  specfem::kokkos::HostView2d<type_real> s_coord("s_coord", 2, ngnod);

  for (int i = 0; i < ngnod; i++) {
    s_coord(0, i) = mesh.control_nodes.h_coord(0, ispec, i);
    s_coord(1, i) = mesh.control_nodes.h_coord(1, ispec, i);
  }

  const auto [x, z] = jacobian::compute_locations(s_coord, ngnod, xi, gamma);

  return { x, z };
}

specfem::point::global_coordinates<specfem::dimension::type::dim2>
specfem::algorithms::locate_point(
    const specfem::kokkos::HostTeam::member_type &team_member,
    const specfem::point::local_coordinates<specfem::dimension::type::dim2>
        &coordinate,
    const specfem::compute::mesh &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.control_nodes.ngnod;

  specfem::kokkos::HostScratchView2d<type_real> s_coord(
      team_member.team_scratch(0), 2, ngnod);

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, ngnod), [&](const int i) {
        s_coord(0, i) = mesh.control_nodes.h_coord(0, ispec, i);
        s_coord(1, i) = mesh.control_nodes.h_coord(1, ispec, i);
      });

  team_member.team_barrier();

  const auto [x, z] =
      jacobian::compute_locations(team_member, s_coord, ngnod, xi, gamma);

  return { x, z };
}
