#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/algorithms/locate_point.hpp"
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

std::pair<type_real, bool> get_local_edge_coordinate(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &global,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const specfem::mesh_entity::dim2::type &mesh_entity, type_real coord) {
  constexpr type_real local_deriv_eps = 1e-12;
  constexpr type_real global_coord_eps = 5e-2;
  const int ngnod = coorg.extent(0);

  // full local coords
  type_real xi, gamma;
  specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true, false>
      jacobian;

  /* coordinate of edge, references either xi or gamma. Other coord is
   * edge-constrained.
   *
   * Additionally, we can reference the coordinate of the jacobian matrix.
   */
  auto [edgecoord, jacobian_edgecoordx, jacobian_edgecoordz] =
      [&xi, &gamma, &mesh_entity,
       &jacobian]() -> std::tuple<type_real &, type_real &, type_real &> {
    if (mesh_entity == specfem::mesh_entity::dim2::type::bottom) {
      gamma = -1;
      return { xi, jacobian.xix, jacobian.xiz };
    } else if (mesh_entity == specfem::mesh_entity::dim2::type::right) {
      xi = 1;
      return { gamma, jacobian.gammax, jacobian.gammaz };
    } else if (mesh_entity == specfem::mesh_entity::dim2::type::top) {
      gamma = 1;
      return { xi, jacobian.xix, jacobian.xiz };
    } else {
      xi = -1;
      return { gamma, jacobian.gammax, jacobian.gammaz };
    }
  }();
  edgecoord = coord;

  for (int iter_loop = 0; iter_loop < 100; iter_loop++) {
    // we may want a dim1 type? for now, just constrain on dim2. update location
    // and jacobian matrix
    auto loc = specfem::jacobian::compute_locations(coorg, ngnod, xi, gamma);
    jacobian = specfem::jacobian::compute_jacobian(coorg, ngnod, xi, gamma);

    type_real dx = -(loc.x - global.x);
    type_real dz = -(loc.z - global.z);

    // step direction:
    type_real dedgecoord = jacobian_edgecoordx * dx + jacobian_edgecoordz * dz;

    // are we on the corner and pointing outside?
    if (edgecoord == -1 && dedgecoord < -local_deriv_eps) {
      return { -1, false };
    }
    if (edgecoord == 1 && dedgecoord > local_deriv_eps) {
      return { 1, false };
    }

    // no out-of-bounds. keep going
    edgecoord += dedgecoord;

    // clamp exactly to bounds
    if (edgecoord > 1) {
      edgecoord = 1;
    } else if (edgecoord < -1) {
      edgecoord = -1;
    }
    // Check for convergence
    if (std::abs(dedgecoord) < local_deriv_eps)
      break;
  }

  // verify point proximity: first get the distance
  auto loc = specfem::jacobian::compute_locations(coorg, ngnod, xi, gamma);
  const type_real distance = specfem::point::distance(global, loc);

  // find some characteristic length. We can use the max diagonal.
  // corner control nodes are always [0,4)
  const type_real mesh_charlen =
      std::max(specfem::point::distance(coorg(0), coorg(2)),
               specfem::point::distance(coorg(1), coorg(3)));

  if (distance > mesh_charlen * global_coord_eps) {
    std::ostringstream oss;
    oss << "\nFailed to locate point along edge:\n"
        << "  (xi, gamma)   = (" << xi << ", " << gamma << ")\n"
        << "  (target_x, target_z) = (" << global.x << ", " << global.z << ")\n"
        << "   (found_x,  found_z) = (" << loc.x << ", " << loc.z << ")\n"
        << "            final_dist = " << distance << "\n"
        << "                             (" << std::scientific
        << distance / mesh_charlen << std::fixed << " x mesh length scale)\n"
        << "This may have been caused by improper meshing along a "
           "nonconforming interface.\n";
    throw std::runtime_error(oss.str());
  }

  return { edgecoord, true };
}

template <typename GraphType>
std::vector<int> get_best_candidates_from_graph(const int ispec_guess,
                                                const GraphType &graph) {

  std::vector<int> ispec_candidates;
  ispec_candidates.push_back(ispec_guess);

  for (auto edge :
       boost::make_iterator_range(boost::out_edges(ispec_guess, graph))) {
    const int ispec = boost::target(edge, graph);
    if (std::find(ispec_candidates.begin(), ispec_candidates.end(), ispec) ==
        ispec_candidates.end()) {
      ispec_candidates.push_back(ispec);
    }
  }
  return ispec_candidates;
}

std::tuple<type_real, type_real> get_best_location(
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

specfem::point::local_coordinates<specfem::dimension::type::dim2>
locate_point_from_best_candidates(
    const std::vector<int> &best_candidates,
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coord,
    const int ngnod) {

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

  return locate_point_from_best_candidates(best_candidates, coordinates,
                                           control_node_coord, ngnod);
}

} // namespace locate_point_impl
} // namespace algorithms
} // namespace specfem
