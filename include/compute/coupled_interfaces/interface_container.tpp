#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/coupled_interfaces/interface_container.hpp"
#include "compute/properties/properties.hpp"
#include "edge/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "point/coordinates.hpp"

namespace {
// Topological map ordering for coupled elements

// +-----------------+      +-----------------+
// |                 |      |                 |
// |               R | ^  ^ | L               |
// |               I | |  | | E               |
// |               G | |  | | F               |
// |               H | |  | | T               |
// |               T | |  | |                 |
// |     BOTTOM      |      |      BOTTOM     |
// +-----------------+      +-----------------+
//   -------------->          -------------->
//   -------------->          -------------->
// +-----------------+      +-----------------+
// |      TOP        |      |       TOP       |
// |               R | ^  ^ | L               |
// |               I | |  | | E               |
// |               G | |  | | F               |
// |               H | |  | | T               |
// |               T | |  | |                 |
// |                 |      |                 |
// +-----------------+      +-----------------+

// // Given an edge, return the range of i, j indices to iterate over the edge
// in
// // correct order The range is normalized to [0,1]
// void get_edge_range(const specfem::enums::edge::type &edge, int &ibegin,
//                     int &jbegin, int &iend, int &jend) {
//   switch (edge) {
//   case specfem::enums::edge::type::BOTTOM:
//     ibegin = 0;
//     jbegin = 0;
//     iend = 1;
//     jend = 0;
//     break;
//   case specfem::enums::edge::type::TOP:
//     ibegin = 1;
//     jbegin = 1;
//     iend = 0;
//     jend = 1;
//     break;
//   case specfem::enums::edge::type::LEFT:
//     ibegin = 0;
//     jbegin = 1;
//     iend = 0;
//     jend = 0;
//     break;
//   case specfem::enums::edge::type::RIGHT:
//     ibegin = 1;
//     jbegin = 0;
//     iend = 1;
//     jend = 1;
//     break;
//   default:
//     throw std::runtime_error("Invalid edge type");
//   }
// }

std::vector<std::tuple<int, int> >
get_points_on_edge(const specfem::enums::edge::type &edge, const int &ngll) {
  std::vector<std::tuple<int, int> > points(ngll);

  switch (edge) {
  case specfem::enums::edge::type::BOTTOM:
    for (int i = 0; i < ngll; i++) {
      points[i] = std::make_tuple(i, 0);
    }
    break;
  case specfem::enums::edge::type::TOP:
    for (int i = 0; i < ngll; i++) {
      points[i] = std::make_tuple(i, ngll - 1);
    }
    break;
  case specfem::enums::edge::type::LEFT:
    for (int i = 0; i < ngll; i++) {
      points[i] = std::make_tuple(0, i);
    }
    break;
  case specfem::enums::edge::type::RIGHT:
    for (int i = 0; i < ngll; i++) {
      points[i] = std::make_tuple(ngll - 1, i);
    }
    break;
  default:
    throw std::runtime_error("Invalid edge type");
  }

  return points;
}
// // Given an edge, return the number of points along the edge
// // This ends up being important when ngllx != ngllz
// KOKKOS_FUNCTION
// int specfem::compute::coupled_interfaces::access::npoints(
//     const specfem::enums::edge::type &edge, const int ngllx, const int ngllz)
//     {

//   switch (edge) {
//   case specfem::enums::edge::type::BOTTOM:
//   case specfem::enums::edge::type::TOP:
//     return ngllx;
//     break;
//   case specfem::enums::edge::type::LEFT:
//   case specfem::enums::edge::type::RIGHT:
//     return ngllz;
//     break;
//   default:
//     assert(false && "Invalid edge type");
//     return 0;
//   }
// }

// KOKKOS_FUNCTION
// void specfem::compute::coupled_interfaces::access::self_iterator(
//     const int &ipoint, const specfem::enums::edge::type &edge, const int
//     ngllx, const int ngllz, int &i, int &j) {

//   switch (edge) {
//   case specfem::enums::edge::type::BOTTOM:
//     i = ipoint;
//     j = 0;
//     break;
//   case specfem::enums::edge::type::TOP:
//     i = ngllx - 1 - ipoint;
//     j = ngllz - 1;
//     break;
//   case specfem::enums::edge::type::LEFT:
//     i = 0;
//     j = ipoint;
//     break;
//   case specfem::enums::edge::type::RIGHT:
//     i = ngllx - 1;
//     j = ngllz - 1 - ipoint;
//     break;
//   default:
//     assert(false && "Invalid edge type");
//   }
// }

// KOKKOS_FUNCTION
// void specfem::compute::coupled_interfaces::access::coupled_iterator(
//     const int &ipoint, const specfem::enums::edge::type &edge, const int
//     ngllx, const int ngllz, int &i, int &j) {

//   switch (edge) {
//   case specfem::enums::edge::type::BOTTOM:
//     i = ngllx - 1 - ipoint;
//     j = 0;
//     break;
//   case specfem::enums::edge::type::TOP:
//     i = ipoint;
//     j = ngllz - 1;
//     break;
//   case specfem::enums::edge::type::LEFT:
//     i = ngllx - 1;
//     j = ngllz - 1 - ipoint;
//     break;
//   case specfem::enums::edge::type::RIGHT:
//     i = 0;
//     j = ipoint;
//     break;
//   default:
//     assert(false && "Invalid edge type");
//   }
// }

bool check_if_edges_are_connected(const specfem::compute::points points,
                                  const specfem::enums::edge::type edge1,
                                  const specfem::enums::edge::type edge2,
                                  const int ispec1, const int ispec2) {

  if (edge1 == edge2) {
    return false;
  }

  const auto h_coord = points.h_coord;

  const int ngll = h_coord.extent(2);

  const auto edge1_points = get_points_on_edge(edge1, ngll);
  const auto edge2_points = get_points_on_edge(edge2, ngll);

  if (edge1_points.size() != edge2_points.size()) {
    throw std::runtime_error(
        "Number of points on edge1 and edge2 are different");
  }

  // size of one edge of the element
  type_real edge_size = [&]() {
    switch (edge1) {
    case specfem::enums::edge::type::BOTTOM:
    case specfem::enums::edge::type::TOP:
      return h_coord(0, ispec1, 0, 1) - h_coord(0, ispec1, 0, 0);
      break;

    case specfem::enums::edge::type::LEFT:
    case specfem::enums::edge::type::RIGHT:
      return h_coord(1, ispec1, 1, 0) - h_coord(1, ispec1, 0, 0);
      break;
    default:
      throw std::runtime_error("Invalid edge type");
    }
  }();

  // Check that the distance every point on the edges is small
  for (int ipoint = 0; ipoint < ngll; ipoint++) {
    const auto [i1, j1] = edge1_points[ipoint];
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        self_coordinates(h_coord(0, ispec1, j1, i1),
                         h_coord(1, ispec1, j1, i1));

    const auto [i2, j2] = edge2_points[ipoint];
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        coupled_coordinates(h_coord(0, ispec2, j2, i2),
                            h_coord(1, ispec2, j2, i2));

    // Check that the distance between the two points is small
    type_real distance =
        specfem::point::distance(self_coordinates, coupled_coordinates);

    if (distance / edge_size > 1.e-10) {
      return false;
    }
  }

  return true;
}

// void compute_edges(
//     const Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
//     h_ibool, const specfem::kokkos::HostMirror1d<int> ispec1, const
//     specfem::kokkos::HostMirror1d<int> ispec2,
//     specfem::kokkos::HostMirror1d<specfem::edge::interface> edge1,
//     specfem::kokkos::HostMirror1d<specfem::edge::interface> edge2) {

//   const int ngll = h_ibool.extent(1);

//   const int num_interfaces = ispec1.extent(0);

//   for (int inum = 0; inum < num_interfaces; inum++) {
//     const int ispec1l = ispec1(inum);
//     const int ispec2l = ispec2(inum);

//     int num_connected = 0;
//     for (int edge1l = 1; edge1l < specfem::enums::edge::num_edges; edge1l++)
//     {
//       for (int edge2l = 1; edge2l < specfem::enums::edge::num_edges;
//       edge2l++) {
//         if (check_if_edges_are_connected(
//                 h_ibool, static_cast<specfem::enums::edge::type>(edge1l),
//                 static_cast<specfem::enums::edge::type>(edge2l), ispec1l,
//                 ispec2l)) {
//           // Check that the two edges are different
//           ASSERT(edge1l != edge2l, "Invalid edge1 and edge2");
//           // BOTTOM-TOP, LEFT-RIGHT coupling
//           ASSERT((((static_cast<specfem::enums::edge::type>(edge1l) ==
//                     specfem::enums::edge::type::BOTTOM) &&
//                    (static_cast<specfem::enums::edge::type>(edge2l) ==
//                     specfem::enums::edge::type::TOP)) ||
//                   ((static_cast<specfem::enums::edge::type>(edge1l) ==
//                     specfem::enums::edge::type::TOP) &&
//                    (static_cast<specfem::enums::edge::type>(edge2l) ==
//                     specfem::enums::edge::type::BOTTOM)) ||
//                   ((static_cast<specfem::enums::edge::type>(edge1l) ==
//                     specfem::enums::edge::type::LEFT) &&
//                    (static_cast<specfem::enums::edge::type>(edge2l) ==
//                     specfem::enums::edge::type::RIGHT)) ||
//                   ((static_cast<specfem::enums::edge::type>(edge1l) ==
//                     specfem::enums::edge::type::RIGHT) &&
//                    (static_cast<specfem::enums::edge::type>(edge2l) ==
//                     specfem::enums::edge::type::LEFT))),
//                  "Invalid edge1 and edge2");

//           edge1(inum) = specfem::edge::interface(
//               static_cast<specfem::enums::edge::type>(edge1l), ngll);
//           edge2(inum) = specfem::edge::interface(
//               static_cast<specfem::enums::edge::type>(edge2l), ngll);
//           num_connected++;
//         }
//       }
//     }
//     // ASSERT(num_connected == 1, "More than one edge is connected");
//   }

//   return;
// }

// void check_edges(
//     const specfem::kokkos::HostView4d<type_real> coordinates,
//     const specfem::kokkos::HostMirror1d<int> ispec1,
//     const specfem::kokkos::HostMirror1d<int> ispec2,
//     const specfem::kokkos::HostMirror1d<specfem::edge::interface> edge1,
//     const specfem::kokkos::HostMirror1d<specfem::edge::interface> edge2) {

//   const int num_interfaces = ispec1.extent(0);
//   const int ngllz = coordinates.extent(2);
//   const int ngllx = coordinates.extent(3);

//   for (int interface = 0; interface < num_interfaces; interface++) {
//     const int ispec1l = ispec1(interface);
//     const int ispec2l = ispec2(interface);

//     const auto edge1l = edge1(interface);
//     const auto edge2l = edge2(interface);

//     // iterate over the edge
//     int npoints = specfem::edge::num_points_on_interface(edge1l);

//     for (int ipoint = 0; ipoint < npoints; ipoint++) {
//       // Get ipoint along the edge in element1
//       int i1, j1;
//       specfem::edge::locate_point_on_self_edge(ipoint, edge1l, j1, i1);
//       const
//       specfem::point::global_coordinates<specfem::dimension::type::dim2>
//       self_coordinates(
//           coordinates(0, ispec1l, j1, i1), coordinates(1, ispec1l, j1, i1));

//       // Get ipoint along the edge in element2
//       int i2, j2;
//       specfem::edge::locate_point_on_coupled_edge(ipoint, edge2l, j2, i2);
//       const
//       specfem::point::global_coordinates<specfem::dimension::type::dim2>
//       coupled_coordinates(
//           coordinates(0, ispec2l, j2, i2), coordinates(1, ispec2l, j2, i2));

//       // Check that the distance between the two points is small

//       type_real distance =
//           specfem::point::distance(self_coordinates, coupled_coordinates);

//       ASSERT(distance < 1.e-10, "Invalid edge1 and edge2");
//     }
//   }
// }

std::tuple<std::vector<type_real>, std::vector<std::array<type_real, 2> > >
compute_edge_factors_and_normals(
    const specfem::compute::points &points,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::quadrature &quadrature, const int ispec1,
    const int ispec2, const specfem::enums::edge::type edge1,
    const specfem::enums::edge::type edge2) {

  const int ngll = points.ngllx;

  const auto edge1_points = get_points_on_edge(edge1, ngll);
  const auto edge2_points = get_points_on_edge(edge2, ngll);

  if (edge1_points.size() != edge2_points.size()) {
    throw std::runtime_error(
        "Number of points on edge1 and edge2 are different");
  }

  std::vector<specfem::point::index<specfem::dimension::type::dim2> >
      medium1_index(ngll);
  std::vector<specfem::point::index<specfem::dimension::type::dim2> >
      medium2_index(ngll);
  std::vector<type_real> edge_factor(ngll);
  std::vector<std::array<type_real, 2> > edge_normal(ngll);

  for (int ipoint = 0; ipoint < ngll; ipoint++) {

    using PointPartialDerivativesType =
        specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                            true, false>;

    const auto [i1, j1] = edge1_points[ipoint];
    const specfem::point::index<specfem::dimension::type::dim2> edge1_index(
        ispec1, j1, i1);
    PointPartialDerivativesType edge1_derivatives;
    specfem::compute::load_on_host(edge1_index, partial_derivatives,
                                   edge1_derivatives);

    const auto [i2, j2] = edge2_points[ipoint];
    const specfem::point::index<specfem::dimension::type::dim2> edge2_index(
        ispec2, j2, i2);
    PointPartialDerivativesType edge2_derivatives;
    specfem::compute::load_on_host(edge2_index, partial_derivatives,
                                   edge2_derivatives);

    const auto edge1_normal = edge1_derivatives.compute_normal(edge1);
    const auto edge2_normal = edge2_derivatives.compute_normal(edge2);

    if ((std::abs(edge1_normal(0) + edge2_normal(0)) >
         1.e-4 * (std::abs(edge1_normal(0) - edge2_normal(0)) / 2)) &&
        (std::abs(edge1_normal(1) + edge2_normal(1)) >
         1.e-4 * (std::abs(edge1_normal(1) - edge2_normal(1)) / 2))) {
      throw std::runtime_error("Edge normals need to be opposite in direction");
    }

    const std::array<type_real, 2> weights = { quadrature.gll.h_weights(i1),
                                               quadrature.gll.h_weights(j1) };

    edge_factor[ipoint] = [&]() {
      switch (edge1) {
      case specfem::enums::edge::type::BOTTOM:
      case specfem::enums::edge::type::TOP:
        return weights[0];
        break;

      case specfem::enums::edge::type::LEFT:
      case specfem::enums::edge::type::RIGHT:
        return weights[1];
        break;
      default:
        throw std::runtime_error("Invalid edge type");
      }
    }();

    edge_normal[ipoint][0] = edge1_normal(0);
    edge_normal[ipoint][1] = edge1_normal(1);
    medium1_index[ipoint] = { ispec1, j1, i1 };
    medium2_index[ipoint] = { ispec2, j2, i2 };
  }

  return { edge_factor, edge_normal };
}

std::tuple<specfem::enums::edge::type, specfem::enums::edge::type,
           std::vector<type_real>, std::vector<std::array<type_real, 2> > >
compute_edge_factors_and_normals(
    const specfem::compute::points &points,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::quadrature &quadrature, const int ispec1,
    const int ispec2) {

  const std::array<specfem::enums::edge::type, 4> edges{
    specfem::enums::edge::type::BOTTOM, specfem::enums::edge::type::TOP,
    specfem::enums::edge::type::LEFT, specfem::enums::edge::type::RIGHT
  };

  std::array<specfem::enums::edge::type, 2> connected_edges;
  int number_of_edges = 0;
  for (const auto edge1 : edges) {
    for (const auto edge2 : edges) {
      if (check_if_edges_are_connected(points, edge1, edge2, ispec1, ispec2)) {
        connected_edges[0] = edge1;
        connected_edges[1] = edge2;
        number_of_edges++;
      }
    }
  }

  if (number_of_edges != 1) {
    std::cout << "Number of edges : " << number_of_edges << std::endl;
    throw std::runtime_error("Number of connected edges is not 1");
  }

  const auto [edge_factor, edge_normal] = compute_edge_factors_and_normals(
      points, partial_derivatives, quadrature, ispec1, ispec2,
      connected_edges[0], connected_edges[1]);

  return { connected_edges[0], connected_edges[1], edge_factor, edge_normal };
}

} // namespace

template <specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2>
specfem::compute::interface_container<
    MediumTag1, MediumTag2>::interface_container(const int num_interfaces,
                                                 const int ngll)
    : num_interfaces(num_interfaces), num_points(ngll),
      medium1_index_mapping(
          "specfem::compute_interface_container::medium1_index_mapping",
          num_interfaces),
      h_medium1_index_mapping(
          Kokkos::create_mirror_view(medium1_index_mapping)),
      medium2_index_mapping(
          "specfem::compute_interface_container::medium2_index_mapping",
          num_interfaces),
      h_medium2_index_mapping(
          Kokkos::create_mirror_view(medium2_index_mapping)),
      medium1_edge_type(
          "specfem::compute_interface_container::medium1_edge_type",
          num_interfaces),
      h_medium1_edge_type(Kokkos::create_mirror_view(medium1_edge_type)),
      medium2_edge_type(
          "specfem::compute_interface_container::medium2_edge_type",
          num_interfaces),
      h_medium2_edge_type(Kokkos::create_mirror_view(medium2_edge_type)),
      medium1_edge_factor(
          "specfem::compute_interface_container::medium1_edge_factor",
          num_interfaces, ngll),
      h_medium1_edge_factor(Kokkos::create_mirror_view(medium1_edge_factor)),
      medium2_edge_factor(
          "specfem::compute_interface_container::medium2_edge_factor",
          num_interfaces, ngll),
      h_medium2_edge_factor(Kokkos::create_mirror_view(medium2_edge_factor)),
      medium1_edge_normal(
          "specfem::compute_interface_container::medium1_edge_normal", 2,
          num_interfaces, ngll),
      h_medium1_edge_normal(Kokkos::create_mirror_view(medium1_edge_normal)),
      medium2_edge_normal(
          "specfem::compute_interface_container::medium2_edge_normal", 2,
          num_interfaces, ngll),
      h_medium2_edge_normal(Kokkos::create_mirror_view(medium2_edge_normal)) {
  return;
}

template <specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2>
specfem::compute::interface_container<MediumTag1, MediumTag2>::
    interface_container(
        const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh, const specfem::compute::points &points,
        const specfem::compute::quadrature &quadratures,
        const specfem::compute::partial_derivatives &partial_derivatives,
        const specfem::compute::element_types &element_types,
        const specfem::compute::mesh_to_compute_mapping &mapping) {

  const auto interface_container =
      std::get<specfem::mesh::interface_container<specfem::dimension::type::dim2, MediumTag1, MediumTag2> >(
          mesh.coupled_interfaces.get<MediumTag1, MediumTag2>());

  int num_interfaces = interface_container.num_interfaces;
  const int ngll = points.ngllx;

  if (num_interfaces == 0) {
    this->num_interfaces = 0;
    return;
  }

  *this = specfem::compute::interface_container<MediumTag1, MediumTag2>(
      num_interfaces, ngll);

  for (int iedge = 0; iedge < num_interfaces; ++iedge) {
    const int ispec1_mesh =
        interface_container.template get_spectral_elem_index<MediumTag1>(iedge);
    const int ispec2_mesh =
        interface_container.template get_spectral_elem_index<MediumTag2>(iedge);

    const int ispec1_compute = mapping.mesh_to_compute(ispec1_mesh);
    const int ispec2_compute = mapping.mesh_to_compute(ispec2_mesh);

    if (!(((element_types.get_medium_tag(ispec1_compute) == MediumTag1) &&
           (element_types.get_medium_tag(ispec2_compute) == MediumTag2)) ||
          ((element_types.get_medium_tag(ispec1_compute) == MediumTag2 &&
            element_types.get_medium_tag(ispec2_compute) == MediumTag1)))) {

      throw std::runtime_error(
          "Coupled Interfaces: Interface is not between the correct mediums");
    }

    h_medium1_index_mapping(iedge) = ispec1_compute;
    h_medium2_index_mapping(iedge) = ispec2_compute;

    const auto [edge1_type, edge2_type, edge_factor, edge_normal] =
        compute_edge_factors_and_normals(points, partial_derivatives,
                                         quadratures, ispec1_compute,
                                         ispec2_compute);

    h_medium1_edge_type(iedge) = edge1_type;
    h_medium2_edge_type(iedge) = edge2_type;

    const int npoints = edge_factor.size();

    for (int ipoint = 0; ipoint < npoints; ipoint++) {
      h_medium1_edge_factor(iedge, ipoint) = edge_factor[ipoint];
      h_medium2_edge_factor(iedge, ipoint) = edge_factor[ipoint];

      h_medium1_edge_normal(0, iedge, ipoint) = edge_normal[ipoint][0];
      h_medium1_edge_normal(1, iedge, ipoint) = edge_normal[ipoint][1];

      h_medium2_edge_normal(0, iedge, ipoint) = -edge_normal[ipoint][0];
      h_medium2_edge_normal(1, iedge, ipoint) = -edge_normal[ipoint][1];
    }
  }

  Kokkos::deep_copy(medium1_index_mapping, h_medium1_index_mapping);
  Kokkos::deep_copy(medium2_index_mapping, h_medium2_index_mapping);
  Kokkos::deep_copy(medium1_edge_type, h_medium1_edge_type);
  Kokkos::deep_copy(medium2_edge_type, h_medium2_edge_type);
  Kokkos::deep_copy(medium1_edge_factor, h_medium1_edge_factor);
  Kokkos::deep_copy(medium2_edge_factor, h_medium2_edge_factor);
  Kokkos::deep_copy(medium1_edge_normal, h_medium1_edge_normal);
  Kokkos::deep_copy(medium2_edge_normal, h_medium2_edge_normal);

  return;
}
