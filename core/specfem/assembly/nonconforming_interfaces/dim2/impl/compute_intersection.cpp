
#include "compute_intersection.hpp"
#include "Kokkos_Core_fwd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/jacobian/dim2/jacobian.hpp"
#include "specfem/point/global_coordinates.hpp"
#include "specfem_setup.hpp"
#include <sstream>
#include <stdexcept>

inline std::pair<
    specfem::point::global_coordinates<specfem::dimension::type::dim2>,
    specfem::point::global_coordinates<specfem::dimension::type::dim2> >
edge_extents(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element_coordinates,
    const specfem::mesh_entity::dim2::type &side) {
  switch (side) {
  case specfem::mesh_entity::dim2::type::bottom:
    // control nodes 0 and 1
    return { element_coordinates(0), element_coordinates(1) };
  case specfem::mesh_entity::dim2::type::right:
    // control nodes 1 and 2
    return { element_coordinates(1), element_coordinates(2) };
  case specfem::mesh_entity::dim2::type::top:
    // control nodes 3 and 2 (order for increasing xi)
    return { element_coordinates(3), element_coordinates(2) };
  case specfem::mesh_entity::dim2::type::left:
    // control nodes 0 and 3 (order for increasing gamma)
    return { element_coordinates(0), element_coordinates(3) };
  default:
    throw std::runtime_error(
        "compute_intersection must be given edges. Found " +
        specfem::mesh_entity::dim2::to_string(side) + ", instead.");
  }
}

std::vector<std::pair<type_real, type_real> >
specfem::assembly::nonconforming_interfaces_impl::compute_intersection(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element1,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element2,
    const specfem::mesh_entity::dim2::type &edge1,
    const specfem::mesh_entity::dim2::type &edge2,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature) {
  constexpr type_real eps = 1e-5;
  const int nquad = mortar_quadrature.extent(0);

  std::vector<std::pair<type_real, type_real> > intersections(nquad);

  // endpoints of edge on either element (local coord = lo:-1,  hi:1)
  const auto [p1_lo, p1_hi] = edge_extents(element1, edge1);
  const auto [p2_lo, p2_hi] = edge_extents(element2, edge2);

  // crossover: find local coordinate of opposite side for each point
  // we do not use the in_element flag.
  const type_real p2lo_on_1 =
      specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
          p2_lo, element1, edge1, 0)
          .first;
  const type_real p2hi_on_1 =
      specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
          p2_hi, element1, edge1, 0)
          .first;
  const type_real p1lo_on_2 =
      specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
          p1_lo, element2, edge2, 0)
          .first;
  const type_real p1hi_on_2 =
      specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
          p1_hi, element2, edge2, 0)
          .first;

  // recover the bounds of the intersection in local coordinates:
  // get_local_edge_coordinate returns values within [-1,1], so an edge's own
  // bounds are already factored in.
  type_real inter_min_on_1 = std::min(p2lo_on_1, p2hi_on_1);
  type_real inter_max_on_1 = std::max(p2lo_on_1, p2hi_on_1);
  type_real inter_min_on_2 = std::min(p1lo_on_2, p1hi_on_2);
  type_real inter_max_on_2 = std::max(p1lo_on_2, p1hi_on_2);

  if (inter_min_on_1 > inter_max_on_1 - eps ||
      inter_min_on_2 > inter_max_on_2 - eps) {
    // no intersection
    std::ostringstream oss;

    specfem::point::global_coordinates<specfem::dimension::type::dim2> center1;
    if (element1.extent(0) == 9) {
      center1 = element1(8);
    } else {
      center1.x =
          (element1(0).x + element1(1).x + element1(2).x + element1(3).x) / 4;
      center1.z =
          (element1(0).z + element1(1).z + element1(2).z + element1(3).z) / 4;
    }

    specfem::point::global_coordinates<specfem::dimension::type::dim2> center2;
    if (element2.extent(0) == 9) {
      center2 = element2(8);
    } else {
      center2.x =
          (element2(0).x + element2(1).x + element2(2).x + element2(3).x) / 4;
      center2.z =
          (element2(0).z + element2(1).z + element2(2).z + element2(3).z) / 4;
    }

    oss << "When computing intersections between element A ("
        << specfem::mesh_entity::dim2::to_string(edge1) << ") and element B ("
        << specfem::mesh_entity::dim2::to_string(edge2)
        << "), no intersection was found.\n"
        << "\n"
        << "Element A center: " << center1 << "\n"
        << "    Edge between: " << p1_lo << " - " << p1_hi << "\n"
        << "    intersection in local coordinates: [" << inter_min_on_1 << ","
        << inter_max_on_1 << "]\n"
        << "Element B center: " << center2 << "\n"
        << "    Edge between: " << p2_lo << " - " << p2_hi << "\n"
        << "    intersection in local coordinates: [" << inter_min_on_2 << ","
        << inter_max_on_2 << "]\n";

    throw std::runtime_error(oss.str());
  }

  /* we may get better accuracy if we properly space out the mortar quadrature
   * points to have |ds| approx constant, but for now just scale points linearly
   * along ispec's local coordinates.
   */
  for (int iquad = 0; iquad < nquad; ++iquad) {
    const type_real xi_mortar = mortar_quadrature(iquad);
    // map from [-1,1] to [inter_min_on_1,inter_max_on_1]
    const type_real xi_i =
        0.5 * ((inter_max_on_1 - inter_min_on_1) * xi_mortar +
               (inter_max_on_1 + inter_min_on_1));
    intersections[iquad].first = xi_i;

    // find corresponding coordinate on jspec through global mapping:

    const auto [xi, gamma] = [&]() -> std::pair<type_real, type_real> {
      if (edge1 == specfem::mesh_entity::dim2::type::bottom) {
        return { xi_i, -1 };
      } else if (edge1 == specfem::mesh_entity::dim2::type::right) {
        return { 1, xi_i };
      } else if (edge1 == specfem::mesh_entity::dim2::type::top) {
        return { xi_i, 1 };
      } else {
        return { -1, xi_i };
      }
    }();

    // map from [-1,1] to [inter_min_on_2,inter_max_on_2]. Recover global coord
    // through xi_i
    intersections[iquad].second =
        specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
            specfem::jacobian::compute_locations(element1, element1.extent(0),
                                                 xi, gamma),
            element2, edge2, 0)
            .first;
  }

  return intersections;
}
