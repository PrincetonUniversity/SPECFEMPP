#ifndef _COMPUTE_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_TPP_
#define _COMPUTE_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_TPP_

#include "compute/coupled_interfaces/interface_container.hpp"
#include "edge/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/coupled_interfaces/coupled_interfaces.hpp"
#include "mesh/coupled_interfaces/interface_container.hpp"
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

// Given an edge, return the range of i, j indices to iterate over the edge in
// correct order The range is normalized to [0,1]
void get_edge_range(const specfem::enums::edge::type &edge, int &ibegin,
                    int &jbegin, int &iend, int &jend) {
  switch (edge) {
  case specfem::enums::edge::type::BOTTOM:
    ibegin = 0;
    jbegin = 0;
    iend = 1;
    jend = 0;
    break;
  case specfem::enums::edge::type::TOP:
    ibegin = 1;
    jbegin = 1;
    iend = 0;
    jend = 1;
    break;
  case specfem::enums::edge::type::LEFT:
    ibegin = 0;
    jbegin = 1;
    iend = 0;
    jend = 0;
    break;
  case specfem::enums::edge::type::RIGHT:
    ibegin = 1;
    jbegin = 0;
    iend = 1;
    jend = 1;
    break;
  default:
    throw std::runtime_error("Invalid edge type");
  }
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

bool check_if_edges_are_connected(
    const Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace> h_ibool,
    const specfem::enums::edge::type &edge1,
    const specfem::enums::edge::type &edge2, const int &ispec1,
    const int &ispec2) {

  // Check that edge1 in element1 is coupling with edge2 in element2
  // The coupling should be in inverse order for the two elements
  // (e.g. BOTTOM-TOP, LEFT-RIGHT)
  // Check the diagram above

  const int ngllx = h_ibool.extent(2);
  const int ngllz = h_ibool.extent(1);

  // Get the range of the two edges
  int ibegin1, jbegin1, iend1, jend1;
  int ibegin2, jbegin2, iend2, jend2;

  get_edge_range(edge1, ibegin1, jbegin1, iend1, jend1);
  get_edge_range(edge2, ibegin2, jbegin2, iend2, jend2);

  // Get the global index range of the two edges
  ibegin1 = ibegin1 * (ngllx - 1);
  iend1 = iend1 * (ngllx - 1);
  jbegin1 = jbegin1 * (ngllz - 1);
  jend1 = jend1 * (ngllz - 1);

  ibegin2 = ibegin2 * (ngllx - 1);
  iend2 = iend2 * (ngllx - 1);
  jbegin2 = jbegin2 * (ngllz - 1);
  jend2 = jend2 * (ngllz - 1);

  // Check if the corners of the two elements have the same global index

  return (h_ibool(ispec1, jbegin1, ibegin1) == h_ibool(ispec2, jend2, iend2)) &&
         (h_ibool(ispec2, jbegin2, ibegin2) == h_ibool(ispec1, jend1, iend1));
}

void compute_edges(
    const Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace> h_ibool,
    const specfem::kokkos::HostMirror1d<int> ispec1,
    const specfem::kokkos::HostMirror1d<int> ispec2,
    specfem::kokkos::HostMirror1d<specfem::edge::interface> edge1,
    specfem::kokkos::HostMirror1d<specfem::edge::interface> edge2) {

  const int ngll = h_ibool.extent(1);

  const int num_interfaces = ispec1.extent(0);

  for (int inum = 0; inum < num_interfaces; inum++) {
    const int ispec1l = ispec1(inum);
    const int ispec2l = ispec2(inum);

    int num_connected = 0;
    for (int edge1l = 1; edge1l < specfem::enums::edge::num_edges; edge1l++) {
      for (int edge2l = 1; edge2l < specfem::enums::edge::num_edges; edge2l++) {
        if (check_if_edges_are_connected(
                h_ibool, static_cast<specfem::enums::edge::type>(edge1l),
                static_cast<specfem::enums::edge::type>(edge2l), ispec1l,
                ispec2l)) {
          // Check that the two edges are different
          ASSERT(edge1l != edge2l, "Invalid edge1 and edge2");
          // BOTTOM-TOP, LEFT-RIGHT coupling
          ASSERT((((static_cast<specfem::enums::edge::type>(edge1l) ==
                    specfem::enums::edge::type::BOTTOM) &&
                   (static_cast<specfem::enums::edge::type>(edge2l) ==
                    specfem::enums::edge::type::TOP)) ||
                  ((static_cast<specfem::enums::edge::type>(edge1l) ==
                    specfem::enums::edge::type::TOP) &&
                   (static_cast<specfem::enums::edge::type>(edge2l) ==
                    specfem::enums::edge::type::BOTTOM)) ||
                  ((static_cast<specfem::enums::edge::type>(edge1l) ==
                    specfem::enums::edge::type::LEFT) &&
                   (static_cast<specfem::enums::edge::type>(edge2l) ==
                    specfem::enums::edge::type::RIGHT)) ||
                  ((static_cast<specfem::enums::edge::type>(edge1l) ==
                    specfem::enums::edge::type::RIGHT) &&
                   (static_cast<specfem::enums::edge::type>(edge2l) ==
                    specfem::enums::edge::type::LEFT))),
                 "Invalid edge1 and edge2");

          edge1(inum) = specfem::edge::interface(
              static_cast<specfem::enums::edge::type>(edge1l), ngll);
          edge2(inum) = specfem::edge::interface(
              static_cast<specfem::enums::edge::type>(edge2l), ngll);
          num_connected++;
        }
      }
    }
    // ASSERT(num_connected == 1, "More than one edge is connected");
  }

  return;
}

void check_edges(
    const specfem::kokkos::HostView4d<type_real> coordinates,
    const specfem::kokkos::HostMirror1d<int> ispec1,
    const specfem::kokkos::HostMirror1d<int> ispec2,
    const specfem::kokkos::HostMirror1d<specfem::edge::interface> edge1,
    const specfem::kokkos::HostMirror1d<specfem::edge::interface> edge2) {

  const int num_interfaces = ispec1.extent(0);
  const int ngllz = coordinates.extent(2);
  const int ngllx = coordinates.extent(3);

  for (int interface = 0; interface < num_interfaces; interface++) {
    const int ispec1l = ispec1(interface);
    const int ispec2l = ispec2(interface);

    const auto edge1l = edge1(interface);
    const auto edge2l = edge2(interface);

    // iterate over the edge
    int npoints = specfem::edge::num_points_on_interface(edge1l);

    for (int ipoint = 0; ipoint < npoints; ipoint++) {
      // Get ipoint along the edge in element1
      int i1, j1;
      specfem::edge::locate_point_on_self_edge(ipoint, edge1l, j1, i1);
      const specfem::point::gcoord2 self_coordinates(
          coordinates(0, ispec1l, j1, i1), coordinates(1, ispec1l, j1, i1));

      // Get ipoint along the edge in element2
      int i2, j2;
      specfem::edge::locate_point_on_coupled_edge(ipoint, edge2l, j2, i2);
      const specfem::point::gcoord2 coupled_coordinates(
          coordinates(0, ispec2l, j2, i2), coordinates(1, ispec2l, j2, i2));

      // Check that the distance between the two points is small

      type_real distance =
          specfem::point::distance(self_coordinates, coupled_coordinates);

      ASSERT(distance < 1.e-10, "Invalid edge1 and edge2");
    }
  }
}
} // namespace

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
specfem::compute::interface_container<medium1, medium2>::interface_container(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties,
    const specfem::mesh::interface_container<medium1, medium2>
        &coupled_interfaces)
    : num_interfaces(coupled_interfaces.num_interfaces),
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
      h_medium2_edge_type(Kokkos::create_mirror_view(medium2_edge_type)) {

  if (num_interfaces == 0) {
    return;
  }

  const auto h_element_types = properties.h_element_types;

  for (int iedge = 0; iedge < num_interfaces; iedge++) {
    const int medium1_spectral_elem_index =
        coupled_interfaces.template get_spectral_elem_index<medium1_type>(
            iedge);
    const int medium2_spectral_elem_index =
        coupled_interfaces.template get_spectral_elem_index<medium2_type>(
            iedge);

    // ASSERT(((medium1_spectral_elem_index != medium2_spectral_elem_index) &&
    //         (h_element_types(medium1_spectral_elem_index) == medium1) &&
    //         (h_element_types(medium2_spectral_elem_index) == medium2)),
    //        "Wrong medium type");

    h_medium1_index_mapping(iedge) = medium1_spectral_elem_index;
    h_medium2_index_mapping(iedge) = medium2_spectral_elem_index;
  }

  const auto index_mapping = mesh.points.h_index_mapping;
  compute_edges(index_mapping, h_medium1_index_mapping, h_medium2_index_mapping,
                h_medium1_edge_type, h_medium2_edge_type);

#ifndef NDEBUG
  const auto coordinates = mesh.points.h_coord;
  check_edges(coordinates, h_medium1_index_mapping, h_medium2_index_mapping,
              h_medium1_edge_type, h_medium2_edge_type);
#endif

  Kokkos::deep_copy(medium1_index_mapping, h_medium1_index_mapping);
  Kokkos::deep_copy(medium2_index_mapping, h_medium2_index_mapping);
  Kokkos::deep_copy(medium1_edge_type, h_medium1_edge_type);
  Kokkos::deep_copy(medium2_edge_type, h_medium2_edge_type);

  return;
}

// template <specfem::element::medium_tag medium1,
//           specfem::element::medium_tag medium2>
// specfem::compute::interface_container<medium1, medium2>::interface_container(
//     const specfem::compute::interface_container<medium2, medium1> &other)
//     : num_interfaces(other.num_interfaces),
//       medium1_index_mapping(other.medium2_index_mapping),
//       h_medium1_index_mapping(other.h_medium2_index_mapping),
//       medium2_index_mapping(other.medium1_index_mapping),
//       h_medium2_index_mapping(other.h_medium1_index_mapping),
//       medium1_edge_type(other.medium2_edge_type),
//       h_medium1_edge_type(other.h_medium2_edge_type),
//       medium2_edge_type(other.medium1_edge_type),
//       h_medium2_edge_type(other.h_medium1_edge_type) {
//   return;
// }

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
template <specfem::element::medium_tag medium>
KOKKOS_INLINE_FUNCTION int specfem::compute::interface_container<
    medium1, medium2>::load_device_index_mapping(const int iedge) const {
  if constexpr (medium == medium1) {
    return medium1_index_mapping(iedge);
  } else if constexpr (medium == medium2) {
    return medium2_index_mapping(iedge);
  } else {
    static_assert("Invalid medium type");
  }
}

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
template <specfem::element::medium_tag medium>
int specfem::compute::interface_container<
    medium1, medium2>::load_host_index_mapping(const int iedge) const {
  if constexpr (medium == medium1) {
    return h_medium1_index_mapping(iedge);
  } else if constexpr (medium == medium2) {
    return h_medium2_index_mapping(iedge);
  } else {
    static_assert("Invalid medium type");
  }
}

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
template <specfem::element::medium_tag medium>
KOKKOS_INLINE_FUNCTION specfem::edge::interface specfem::compute::
    interface_container<medium1, medium2>::load_device_edge_type(
        const int iedge) const {
  if constexpr (medium == medium1) {
    return medium1_edge_type(iedge);
  } else if constexpr (medium == medium2) {
    return medium2_edge_type(iedge);
  } else {
    static_assert("Invalid medium type");
  }
}

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
template <specfem::element::medium_tag medium>
specfem::edge::interface specfem::compute::interface_container<
    medium1, medium2>::load_host_edge_type(const int iedge) const {
  if constexpr (medium == medium1) {
    return h_medium1_edge_type(iedge);
  } else if constexpr (medium == medium2) {
    return h_medium2_edge_type(iedge);
  } else {
    static_assert("Invalid medium type");
  }
}

#endif /* _COMPUTE_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_TPP_ */
