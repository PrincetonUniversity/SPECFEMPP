#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/algorithms/inside_outside.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <sstream>
#include <stdexcept>

namespace specfem {
namespace algorithms {
namespace locate_point_impl {

// 3D implementations moved from anonymous namespace

std::tuple<int, int, int, int>
rough_location(const specfem::point::global_coordinates<
                   specfem::element::dimension_tag::dim3> &global,
               const MeshHostCoordinatesViewType3D coord) {

  const int nspec = coord.extent(0);
  const int ngllz = coord.extent(1);
  const int nglly = coord.extent(2);
  const int ngllx = coord.extent(3);

  type_real dist_min = std::numeric_limits<type_real>::max();
  int ispec_selected, ix_selected, iy_selected, iz_selected;

  // Nearest-GLL-point search over the whole mesh: O(nspec * ngll^3) per point.
  // This is a linear scan, wasteful on a large globe mesh, but sources and
  // receivers are few and located once at setup, so the cost is tolerable. It
  // only seeds the initial guess; the adjacency-restricted Newton search
  // (get_best_candidates + get_local_coordinates) keeps the expensive inverse
  // mapping to the handful of elements around this guess.
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int k = 0; k < ngllz; k++) {
      for (int j = 0; j < nglly; j++) {
        for (int i = 0; i < ngllx; i++) {

          // Get the global coordinates of the quadrature point
          const specfem::point::global_coordinates<
              specfem::element::dimension_tag::dim3>
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

std::tuple<type_real, type_real, type_real> get_local_coordinates(
    const Kokkos::View<specfem::point::global_coordinates<
                           specfem::element::dimension_tag::dim3> *,
                       Kokkos::HostSpace> &coorg_shifted,
    type_real xi, type_real eta, type_real gamma) {

  const int ngnod = coorg_shifted.extent(0);

  // Newton inverse mapping (physical -> reference) for a curved hex27 element
  // with the target at the origin. The anchors are pre-shifted by the target,
  // so the residual is element-sized in float. The loop stops on the
  // reference-coordinate step, which is unit-free.
  constexpr int max_iterations = 8;
  constexpr type_real step_tolerance = 1e-6;
  type_real d_min_sq = std::numeric_limits<type_real>::max();

  for (int iter_loop = 0; iter_loop < max_iterations; iter_loop++) {
    auto loc = specfem::jacobian::compute_locations(coorg_shifted, ngnod, xi,
                                                    eta, gamma);
    auto jacobian = specfem::jacobian::compute_jacobian(coorg_shifted, ngnod,
                                                        xi, eta, gamma);

    const type_real dx = -loc.x;
    const type_real dy = -loc.y;
    const type_real dz = -loc.z;

    const type_real d_sq = dx * dx + dy * dy + dz * dz;

    if (d_sq < d_min_sq) {
      d_min_sq = d_sq;
    } else {
      break;
    }

    type_real dxi =
        jacobian.xix() * dx + jacobian.xiy() * dy + jacobian.xiz() * dz;
    type_real deta =
        jacobian.etax() * dx + jacobian.etay() * dy + jacobian.etaz() * dz;
    type_real dgamma = jacobian.gammax() * dx + jacobian.gammay() * dy +
                       jacobian.gammaz() * dz;

    // decreases step length if step is large
    if ((dxi * dxi + deta * deta + dgamma * dgamma) >
        static_cast<type_real>(1.0)) {
      constexpr type_real scale = 0.33333333333;
      dxi *= scale;
      deta *= scale;
      dgamma *= scale;
    }

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

    if (dxi * dxi + deta * deta + dgamma * dgamma <
        step_tolerance * step_tolerance)
      break;
  }

  return std::make_tuple(xi, eta, gamma);
}

// Core locate_point logic extracted for testability
specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>
locate_point_core(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const MeshHostCoordinatesViewType3D &global_coordinates,
    const Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &index_mapping,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coordinates,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &gll_points,
    const int ngnod, const int ngllx) {

  int ix_guess, iy_guess, iz_guess, ispec_guess;

  std::tie(ispec_guess, ix_guess, iy_guess, iz_guess) =
      rough_location(coordinates, global_coordinates);

  const auto best_candidates = get_best_candidates(ispec_guess, index_mapping);

  type_real final_dist = std::numeric_limits<type_real>::max();
  bool best_inside = false;

  int ispec_selected_point = -1;
  type_real xi_selected = -9999.0;
  type_real eta_selected = -9999.0;
  type_real gamma_selected = -9999.0;
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      coord_point;

  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim3> *,
                     Kokkos::HostSpace>
      coorg_shifted("coorg_shifted", ngnod);

  const int ngllz = global_coordinates.extent(1);
  const int nglly = global_coordinates.extent(2);

  for (auto &ispec : best_candidates) {
    for (int i = 0; i < ngnod; i++) {
      coorg_shifted(i).x =
          control_node_coordinates(ispec, i, 0) - coordinates.x;
      coorg_shifted(i).y =
          control_node_coordinates(ispec, i, 1) - coordinates.y;
      coorg_shifted(i).z =
          control_node_coordinates(ispec, i, 2) - coordinates.z;
    }

    // Start Newton at the GLL point of this element nearest to the target
    type_real nearest_sq = std::numeric_limits<type_real>::max();
    int nearest_i = 0, nearest_j = 0, nearest_k = 0;
    for (int k = 0; k < ngllz; k++) {
      for (int j = 0; j < nglly; j++) {
        for (int i = 0; i < ngllx; i++) {
          const type_real dx =
              global_coordinates(ispec, k, j, i, 0) - coordinates.x;
          const type_real dy =
              global_coordinates(ispec, k, j, i, 1) - coordinates.y;
          const type_real dz =
              global_coordinates(ispec, k, j, i, 2) - coordinates.z;
          const type_real dist_sq = dx * dx + dy * dy + dz * dz;
          if (dist_sq < nearest_sq) {
            nearest_sq = dist_sq;
            nearest_i = i;
            nearest_j = j;
            nearest_k = k;
          }
        }
      }
    }

    type_real xi_guess = gll_points(nearest_i);
    type_real eta_guess = gll_points(nearest_j);
    type_real gamma_guess = gll_points(nearest_k);

    // Find the best location using Newton-Raphson
    std::tie(xi_guess, eta_guess, gamma_guess) =
        get_local_coordinates(coorg_shifted, xi_guess, eta_guess, gamma_guess);

    // Back-project in the shifted frame; the target is the origin
    const auto loc_shifted = specfem::jacobian::compute_locations(
        coorg_shifted, ngnod, xi_guess, eta_guess, gamma_guess);
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3>
        cart_coord = { coordinates.x + loc_shifted.x,
                       coordinates.y + loc_shifted.y,
                       coordinates.z + loc_shifted.z };

    const type_real dist = std::sqrt(loc_shifted.x * loc_shifted.x +
                                     loc_shifted.y * loc_shifted.y +
                                     loc_shifted.z * loc_shifted.z);

    const type_real diagonal =
        specfem::point::distance(coorg_shifted(0), coorg_shifted(6));
    const type_real equal_tolerance = 1e-5 * diagonal;
    const type_real close_tolerance = 1e-3 * diagonal;
    const bool new_inside =
        specfem::algorithms::inside(specfem::point::local_coordinates<
                                    specfem::element::dimension_tag::dim3>(
            ispec, xi_guess, eta_guess, gamma_guess));

    bool better =
        ispec_selected_point < 0 || dist < final_dist - equal_tolerance;
    if (ispec_selected_point >= 0) {
      if (new_inside && !best_inside && dist < close_tolerance)
        better = true;
      if (!new_inside && best_inside && dist > final_dist - equal_tolerance)
        better = false;
    }

    if (better) {
      ispec_selected_point = ispec;
      xi_selected = xi_guess;
      eta_selected = eta_guess;
      gamma_selected = gamma_guess;
      coord_point = cart_coord;
      final_dist = dist;
      best_inside = new_inside;
    }
  }

  // Check if the found coordinates are valid
  bool xi_out_of_bounds = std::fabs(std::fabs(xi_selected) - 1.01) < 1e-6;
  bool eta_out_of_bounds = std::fabs(std::fabs(eta_selected) - 1.01) < 1e-6;
  bool gamma_out_of_bounds = std::fabs(std::fabs(gamma_selected) - 1.01) < 1e-6;
  bool ispec_invalid = ispec_selected_point < 0;

  // If the found coordinates are out of bounds, throw an error
  if (xi_out_of_bounds || eta_out_of_bounds || gamma_out_of_bounds ||
      ispec_invalid) {
    std::ostringstream oss;
    oss << "\nFailed to locate point in the mesh:\n"
        << "  (ispec, xi, eta, gamma) = (" << ispec_selected_point << ", "
        << xi_selected << ", " << eta_selected << ", " << gamma_selected
        << ")\n"
        << "  (target_x, target_y, target_z) = (" << coordinates.x << ", "
        << coordinates.y << ", " << coordinates.z << ")\n"
        << "   (found_x,  found_y,  found_z) = (" << coord_point.x << ", "
        << coord_point.y << ", " << coord_point.z << ")\n"
        << "            final_dist = " << final_dist << "\n";
    throw std::runtime_error(oss.str());
  }

  return { ispec_selected_point, xi_selected, eta_selected, gamma_selected };
}

specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {
  return locate_point_core(coordinates, mesh.h_coord, mesh.h_index_mapping,
                           mesh.h_control_node_coordinates, mesh.h_xi,
                           mesh.ngnod, mesh.element_grid.ngllx);
}

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinate,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {
  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real eta = coordinate.eta;
  const type_real gamma = coordinate.gamma;
  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::element::dimension_tag::dim3> *,
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
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type
        &team_member,
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinate,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {
  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real eta = coordinate.eta;
  const type_real gamma = coordinate.gamma;
  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::element::dimension_tag::dim3> *,
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

std::pair<std::pair<type_real, type_real>, bool> get_local_face_coordinate(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &global,
    const Kokkos::View<specfem::point::global_coordinates<
                           specfem::element::dimension_tag::dim3> *,
                       Kokkos::HostSpace> &coorg,
    const specfem::mesh_entity::dim3::type &mesh_entity,
    std::pair<type_real, type_real> coord) {
  constexpr type_real local_deriv_eps = 1e-12;
  constexpr type_real global_coord_eps = 5e-2;
  const int ngnod = coorg.extent(0);

  // full local coords
  type_real xi, gamma, eta;
  specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3, true,
                                  false>
      jacobian;

  /* coordinate of edge, references either xi or gamma. Other coord is
   * edge-constrained.
   *
   * Additionally, we can reference the coordinate of the jacobian matrix.
   */
  auto [facecoord1, facecoord2, jacobian_facecoord1x, jacobian_facecoord1y,
        jacobian_facecoord1z, jacobian_facecoord2x, jacobian_facecoord2y,
        jacobian_facecoord2z] = [&xi, &gamma, &eta, &mesh_entity, &jacobian]()
      -> std::tuple<type_real &, type_real &, type_real &, type_real &,
                    type_real &, type_real &, type_real &, type_real &> {
    // Each face pair (bottom/top, left/right, front/back) frees the same two
    // reference coordinates and constrains the third to -1 or +1; only that
    // sign differs within a pair.
    if (mesh_entity == specfem::mesh_entity::dim3::type::bottom ||
        mesh_entity == specfem::mesh_entity::dim3::type::top) {
      gamma = (mesh_entity == specfem::mesh_entity::dim3::type::top) ? 1 : -1;
      return { xi,
               eta,
               jacobian.xix(),
               jacobian.xiy(),
               jacobian.xiz(),
               jacobian.etax(),
               jacobian.etay(),
               jacobian.etaz() };
    } else if (mesh_entity == specfem::mesh_entity::dim3::type::left ||
               mesh_entity == specfem::mesh_entity::dim3::type::right) {
      xi = (mesh_entity == specfem::mesh_entity::dim3::type::right) ? 1 : -1;
      return { eta,
               gamma,
               jacobian.etax(),
               jacobian.etay(),
               jacobian.etaz(),
               jacobian.gammax(),
               jacobian.gammay(),
               jacobian.gammaz() };
    } else { // front / back
      eta = (mesh_entity == specfem::mesh_entity::dim3::type::back) ? 1 : -1;
      return { xi,
               gamma,
               jacobian.xix(),
               jacobian.xiy(),
               jacobian.xiz(),
               jacobian.gammax(),
               jacobian.gammay(),
               jacobian.gammaz() };
    }
  }();
  facecoord1 = coord.first;
  facecoord2 = coord.second;

  for (int iter_loop = 0; iter_loop < 100; iter_loop++) {
    // we may want a dim1 type? for now, just constrain on dim2. update location
    // and jacobian matrix
    auto loc =
        specfem::jacobian::compute_locations(coorg, ngnod, xi, eta, gamma);
    jacobian =
        specfem::jacobian::compute_jacobian(coorg, ngnod, xi, eta, gamma);

    type_real dx = -(loc.x - global.x);
    type_real dy = -(loc.y - global.y);
    type_real dz = -(loc.z - global.z);

    // step direction:
    type_real dfacecoord1 = jacobian_facecoord1x * dx +
                            jacobian_facecoord1y * dy +
                            jacobian_facecoord1z * dz;
    type_real dfacecoord2 = jacobian_facecoord2x * dx +
                            jacobian_facecoord2y * dy +
                            jacobian_facecoord2z * dz;

    // are we on the edge and pointing outside?
    bool facecoord1_on_edge =
        (facecoord1 == -1 && dfacecoord1 < -local_deriv_eps) ||
        (facecoord1 == 1 && dfacecoord1 > local_deriv_eps);
    bool facecoord2_on_edge =
        (facecoord2 == -1 && dfacecoord2 < -local_deriv_eps) ||
        (facecoord2 == 1 && dfacecoord2 > local_deriv_eps);

    // out of bounds check: if (we are on edge and minimized along tangential
    // coordinate) or (we are on corner)?
    if ((facecoord1_on_edge &&
         (facecoord2_on_edge || std::fabs(dfacecoord2) < local_deriv_eps)) ||
        (facecoord2_on_edge && std::fabs(dfacecoord1) < local_deriv_eps)) {
      return { { facecoord1, facecoord2 }, false };
    }

    // no out-of-bounds. keep going
    facecoord1 += dfacecoord1;
    facecoord2 += dfacecoord2;

    // clamp exactly to bounds
    if (facecoord1 > 1) {
      facecoord1 = 1;
    } else if (facecoord1 < -1) {
      facecoord1 = -1;
    }
    if (facecoord2 > 1) {
      facecoord2 = 1;
    } else if (facecoord2 < -1) {
      facecoord2 = -1;
    }

    // Check for convergence
    if (dfacecoord1 * dfacecoord1 + dfacecoord2 * dfacecoord2 <
        local_deriv_eps * local_deriv_eps)
      break;
  }

  // verify point proximity: first get the distance
  auto loc = specfem::jacobian::compute_locations(coorg, ngnod, xi, eta, gamma);
  const type_real distance = specfem::point::distance(global, loc);

  const type_real mesh_charlen = std::max(
      specfem::point::distance(coorg(2), coorg(4)),
      std::max(specfem::point::distance(coorg(0), coorg(6)),
               std::max(specfem::point::distance(coorg(1), coorg(7)),
                        specfem::point::distance(coorg(3), coorg(5)))));

  if (distance > mesh_charlen * global_coord_eps) {
    std::ostringstream oss;
    oss << "\nFailed to locate point along face:\n"
        << "  (xi, eta, gamma)   = (" << xi << ", " << eta << ", " << gamma
        << ")\n"
        << "  (target_x, target_y, target_z) = (" << global.x << ", "
        << global.y << ", " << global.z << ")\n"
        << "   (found_x, found_y,  found_z) = (" << loc.x << ", " << loc.y
        << ", " << loc.z << ")\n"
        << "            final_dist = " << distance << "\n"
        << "                             (" << std::scientific
        << distance / mesh_charlen << std::fixed << " x mesh length scale)\n"
        << "This may have been caused by improper meshing along a "
           "nonconforming interface.\n";
    throw std::runtime_error(oss.str());
  }

  return { { facecoord1, facecoord2 }, true };
}

std::pair<std::pair<type_real, type_real>, bool> locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const int &ispec,
    const specfem::mesh_entity::type<specfem::element::dimension_tag::dim3>
        &constraint) {

  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                      constraint)) {
    throw std::runtime_error("locate_point_on_face constraint must be a face "
                             "(edges are currently unsupported).");
    return { { 0, 0 }, false };
  }
  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim3> *,
                     Kokkos::HostSpace>
      coorg("coorg", mesh.ngnod);
  for (int i = 0; i < mesh.ngnod; i++) {
    coorg(i).x = mesh.h_control_node_coordinates(0, ispec, i);
    coorg(i).y = mesh.h_control_node_coordinates(1, ispec, i);
    coorg(i).z = mesh.h_control_node_coordinates(2, ispec, i);
  }

  // initial guess of 0 (center of edge)
  return specfem::algorithms::locate_point_impl::get_local_face_coordinate(
      coordinates, coorg, constraint, { 0, 0 });
}

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const std::pair<type_real, type_real> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const int &ispec,
    const specfem::mesh_entity::type<specfem::element::dimension_tag::dim3>
        &constraint) {
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                      constraint)) {
    throw std::runtime_error("locate_point_on_face constraint must be a face "
                             "(edges are currently unsupported).");
    return { 0, 0, 0 };
  }
  const auto [xi, gamma,
              eta] = [&]() -> std::tuple<type_real, type_real, type_real> {
    if (constraint == specfem::mesh_entity::dim3::type::bottom) {
      return { coordinates.first, coordinates.second, -1 };
    } else if (constraint == specfem::mesh_entity::dim3::type::right) {
      return { 1, coordinates.first, coordinates.second };
    } else if (constraint == specfem::mesh_entity::dim3::type::top) {
      return { coordinates.first, coordinates.second, 1 };
    } else if (constraint == specfem::mesh_entity::dim3::type::left) {
      return { -1, coordinates.first, coordinates.second };
    } else if (constraint == specfem::mesh_entity::dim3::type::front) {
      return { coordinates.first, -1, coordinates.second };
    } else { // back
      return { coordinates.first, 1, coordinates.second };
    }
  }();

  // interpolating the entire element is not the most efficient way to do this.
  // consider a codimension 1 interpolation in the future.
  return locate_point(
      specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>(
          ispec, xi, eta, gamma),
      mesh);
}

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh_entity::dim3::type &constraint) {
  constexpr type_real eps = 1e-5;
  switch (constraint) {
  case mesh_entity::dim3::type::bottom:
    if (std::abs(coordinates.gamma + 1) > eps) {
      throw std::runtime_error(
          "locate_point constraint and element-local "
          "coordinates provided, but disagree. (bottom, but gamma != -1");
    }
    break;
  case mesh_entity::dim3::type::right:
    if (std::abs(coordinates.xi - 1) > eps) {
      throw std::runtime_error(
          "locate_point constraint and element-local "
          "coordinates provided, but disagree. (right, but xi != 1");
    }
    break;
  case mesh_entity::dim3::type::top:
    if (std::abs(coordinates.gamma - 1) > eps) {
      throw std::runtime_error(
          "locate_point constraint and element-local "
          "coordinates provided, but disagree. (top, but gamma != 1");
    }
    break;
  case mesh_entity::dim3::type::left:
    if (std::abs(coordinates.xi + 1) > eps) {
      throw std::runtime_error(
          "locate_point constraint and element-local "
          "coordinates provided, but disagree. (left, but xi != -1");
    }
    break;
  case mesh_entity::dim3::type::front:
    if (std::abs(coordinates.eta + 1) > eps) {
      throw std::runtime_error(
          "locate_point constraint and element-local "
          "coordinates provided, but disagree. (front, but eta != -1");
    }
    break;
  case mesh_entity::dim3::type::back:
    if (std::abs(coordinates.eta - 1) > eps) {
      throw std::runtime_error(
          "locate_point constraint and element-local "
          "coordinates provided, but disagree. (back, but eta != 1");
    }
    break;
  default:
    throw std::runtime_error("locate_point_on_face constraint must be a face "
                             "(edges are currently unsupported).");
    break;
  }
  return locate_point(coordinates, mesh);
}

} // namespace locate_point_impl
} // namespace algorithms
} // namespace specfem
