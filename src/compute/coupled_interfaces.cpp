// #include "compute/coupled_interfaces.hpp"
// #include "compute/coupled_interfaces.tpp"
#include "compute/coupled_interfaces/coupled_interfaces.hpp"
#include "compute/coupled_interfaces/coupled_interfaces.tpp"
#include "compute/coupled_interfaces/interface_container.hpp"
#include "compute/coupled_interfaces/interface_container.tpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "mesh/coupled_interfaces/coupled_interfaces.hpp"

// // Topological map ordering for coupled elements

// // +-----------------+      +-----------------+
// // |                 |      |                 |
// // |               R | ^  ^ | L               |
// // |               I | |  | | E               |
// // |               G | |  | | F               |
// // |               H | |  | | T               |
// // |               T | |  | |                 |
// // |     BOTTOM      |      |      BOTTOM     |
// // +-----------------+      +-----------------+
// //   -------------->          -------------->
// //   -------------->          -------------->
// // +-----------------+      +-----------------+
// // |      TOP        |      |       TOP       |
// // |               R | ^  ^ | L               |
// // |               I | |  | | E               |
// // |               G | |  | | F               |
// // |               H | |  | | T               |
// // |               T | |  | |                 |
// // |                 |      |                 |
// // +-----------------+      +-----------------+

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

// bool check_if_edges_are_connected(
//     const specfem::kokkos::HostView3d<int> h_ibool,
//     const specfem::enums::edge::type &edge1,
//     const specfem::enums::edge::type &edge2, const int &ispec1,
//     const int &ispec2) {

//   // Check that edge1 in element1 is coupling with edge2 in element2
//   // The coupling should be in inverse order for the two elements
//   // (e.g. BOTTOM-TOP, LEFT-RIGHT)
//   // Check the diagram above

//   const int ngllx = h_ibool.extent(2);
//   const int ngllz = h_ibool.extent(1);

//   // Get the range of the two edges
//   int ibegin1, jbegin1, iend1, jend1;
//   int ibegin2, jbegin2, iend2, jend2;

//   get_edge_range(edge1, ibegin1, jbegin1, iend1, jend1);
//   get_edge_range(edge2, ibegin2, jbegin2, iend2, jend2);

//   // Get the global index range of the two edges
//   ibegin1 = ibegin1 * (ngllx - 1);
//   iend1 = iend1 * (ngllx - 1);
//   jbegin1 = jbegin1 * (ngllz - 1);
//   jend1 = jend1 * (ngllz - 1);

//   ibegin2 = ibegin2 * (ngllx - 1);
//   iend2 = iend2 * (ngllx - 1);
//   jbegin2 = jbegin2 * (ngllz - 1);
//   jend2 = jend2 * (ngllz - 1);

//   // Check if the corners of the two elements have the same global index

//   return (h_ibool(ispec1, jbegin1, ibegin1) == h_ibool(ispec2, jend2, iend2))
//   &&
//          (h_ibool(ispec2, jbegin2, ibegin2) == h_ibool(ispec1, jend1,
//          iend1));
// }

// void compute_edges(
//     const specfem::kokkos::HostMirror3d<int> h_ibool,
//     const specfem::kokkos::HostMirror1d<int> ispec1,
//     const specfem::kokkos::HostMirror1d<int> ispec2,
//     specfem::kokkos::HostMirror1d<specfem::enums::edge::type> edge1,
//     specfem::kokkos::HostMirror1d<specfem::enums::edge::type> edge2) {

//   const int num_interfaces = ispec1.extent(0);

//   for (int inum = 0; inum < num_interfaces; inum++) {
//     const int ispec1l = ispec1(inum);
//     const int ispec2l = ispec2(inum);

//     int num_connected = 0;
//     for (int edge1l = 0; edge1l < specfem::enums::edge::num_edges; edge1l++)
//     {
//       for (int edge2l = 0; edge2l < specfem::enums::edge::num_edges;
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

//           edge1(inum) = static_cast<specfem::enums::edge::type>(edge1l);
//           edge2(inum) = static_cast<specfem::enums::edge::type>(edge2l);
//           num_connected++;
//         }
//       }
//     }
//     ASSERT(num_connected == 1, "More than one edge is connected");
//   }

//   return;
// }

// void check_edges(
//     const specfem::kokkos::HostMirror3d<int> h_ibool,
//     const specfem::kokkos::HostView2d<type_real> coord,
//     const specfem::kokkos::HostMirror1d<int> ispec1,
//     const specfem::kokkos::HostMirror1d<int> ispec2,
//     const specfem::kokkos::HostMirror1d<specfem::enums::edge::type> edge1,
//     const specfem::kokkos::HostMirror1d<specfem::enums::edge::type> edge2) {

//   const int num_interfaces = ispec1.extent(0);
//   const int ngllx = h_ibool.extent(2);
//   const int ngllz = h_ibool.extent(1);

//   for (int interface = 0; interface < num_interfaces; interface++) {
//     const int ispec1l = ispec1(interface);
//     const int ispec2l = ispec2(interface);

//     const auto edge1l = edge1(interface);
//     const auto edge2l = edge2(interface);

//     // iterate over the edge
//     int npoints = specfem::compute::coupled_interfaces::access::npoints(
//         edge1l, ngllx, ngllz);

//     for (int ipoint = 0; ipoint < npoints; ipoint++) {
//       // Get ipoint along the edge in element1
//       int i1, j1;
//       specfem::compute::coupled_interfaces::access::self_iterator(
//           ipoint, edge1l, ngllx, ngllz, i1, j1);
//       const int iglob1 = h_ibool(ispec1l, j1, i1);

//       // Get ipoint along the edge in element2
//       int i2, j2;
//       specfem::compute::coupled_interfaces::access::coupled_iterator(
//           ipoint, edge2l, ngllx, ngllz, i2, j2);
//       const int iglob2 = h_ibool(ispec2l, j2, i2);

//       // Check that the distance between the two points is small

//       type_real distance = (((coord(0, iglob1) - coord(0, iglob2)) *
//                              (coord(0, iglob1) - coord(0, iglob2))) +
//                             ((coord(1, iglob1) - coord(1, iglob2)) *
//                              (coord(1, iglob1) - coord(1, iglob2))));

//       ASSERT((((coord(0, iglob1) - coord(0, iglob2)) *
//                (coord(0, iglob1) - coord(0, iglob2))) +
//               ((coord(1, iglob1) - coord(1, iglob2)) *
//                (coord(1, iglob1) - coord(1, iglob2)))) < 1.e-10,
//              "Invalid edge1 and edge2");
//     }
//   }
// }

// specfem::compute::coupled_interfaces::elastic_acoustic::elastic_acoustic(
//     const specfem::kokkos::HostMirror3d<int> h_ibool,
//     const specfem::kokkos::HostView2d<type_real> coord,
//     const specfem::kokkos::HostView1d<specfem::enums::element::type>
//         h_ispec_type,
//     const specfem::mesh::coupled_interfaces::elastic_acoustic
//     &elastic_acoustic) : num_interfaces(elastic_acoustic.num_interfaces),
//       elastic_ispec(
//           "compute::coupled_interfaces::elastic_acoustic::elastic_ispec",
//           elastic_acoustic.num_interfaces),
//       acoustic_ispec(
//           "compute::coupled_interfaces::elastic_acoustic::acoustic_ispec",
//           elastic_acoustic.num_interfaces),
//       elastic_edge(
//           "compute::coupled_interfaces::elastic_acoustic::elastic_edge",
//           elastic_acoustic.num_interfaces),
//       acoustic_edge(
//           "compute::coupled_interfaces::elastic_acoustic::acoustic_edge",
//           elastic_acoustic.num_interfaces) {

//   if (num_interfaces == 0)
//     return;

//   h_elastic_ispec = Kokkos::create_mirror_view(elastic_ispec);
//   h_acoustic_ispec = Kokkos::create_mirror_view(acoustic_ispec);
//   h_elastic_edge = Kokkos::create_mirror_view(elastic_edge);
//   h_acoustic_edge = Kokkos::create_mirror_view(acoustic_edge);

//   h_elastic_ispec = elastic_acoustic.elastic_ispec;
//   h_acoustic_ispec = elastic_acoustic.acoustic_ispec;

//   ASSERT(h_elastic_ispec.extent(0) == num_interfaces, "Invalid
//   elastic_ispec"); ASSERT(h_acoustic_ispec.extent(0) == num_interfaces,
//          "Invalid acoustic_ispec");
//   ASSERT(elastic_edge.extent(0) == num_interfaces, "Invalid elastic_edge");
//   ASSERT(h_acoustic_edge.extent(0) == num_interfaces, "Invalid
//   acoustic_edge");

// #ifndef NDEBUG
//   for (int i = 0; i < num_interfaces; i++) {
//     int ispec_elastic = h_elastic_ispec(i);
//     int ispec_acoustic = h_acoustic_ispec(i);

//     // Check that the interface is between an elastic and an acoustic element

//     ASSERT(((ispec_elastic != ispec_acoustic) &&
//             (h_ispec_type(ispec_elastic) ==
//              specfem::element::medium_tag::elastic_sv) &&
//             (h_ispec_type(ispec_acoustic) ==
//              specfem::element::medium_tag::acoustic)),
//            "Invalid interface");
//   }
// #endif

//   compute_edges(h_ibool, h_elastic_ispec, h_acoustic_ispec, h_elastic_edge,
//                 h_acoustic_edge);

// #ifndef NDEBUG
//   check_edges(h_ibool, coord, h_elastic_ispec, h_acoustic_ispec,
//   h_elastic_edge,
//               h_acoustic_edge);
// #endif

//   Kokkos::deep_copy(elastic_ispec, h_elastic_ispec);
//   Kokkos::deep_copy(acoustic_ispec, h_acoustic_ispec);
//   Kokkos::deep_copy(elastic_edge, h_elastic_edge);
//   Kokkos::deep_copy(acoustic_edge, h_acoustic_edge);

//   return;
// }

// specfem::compute::coupled_interfaces::elastic_poroelastic::elastic_poroelastic(
//     const specfem::kokkos::HostMirror3d<int> h_ibool,
//     const specfem::kokkos::HostView2d<type_real> coord,
//     const specfem::kokkos::HostView1d<specfem::enums::element::type>
//         h_ispec_type,
//     const specfem::mesh::coupled_interfaces::elastic_poroelastic
//         &elastic_poroelastic)
//     : num_interfaces(elastic_poroelastic.num_interfaces),
//       elastic_ispec(
//           "compute::coupled_interfaces::elastic_poroelastic::elastic_ispec",
//           elastic_poroelastic.num_interfaces),
//       poroelastic_ispec("compute::coupled_interfaces::elastic_poroelastic::"
//                         "poroelastic_ispec",
//                         elastic_poroelastic.num_interfaces),
//       elastic_edge(
//           "compute::coupled_interfaces::elastic_poroelastic::elastic_edge",
//           elastic_poroelastic.num_interfaces),
//       poroelastic_edge("compute::coupled_interfaces::elastic_poroelastic::"
//                        "poroelastic_edge",
//                        elastic_poroelastic.num_interfaces) {

//   if (num_interfaces == 0)
//     return;

//   h_elastic_ispec = Kokkos::create_mirror_view(elastic_ispec);
//   h_poroelastic_ispec = Kokkos::create_mirror_view(poroelastic_ispec);
//   h_elastic_edge = Kokkos::create_mirror_view(elastic_edge);
//   h_poroelastic_edge = Kokkos::create_mirror_view(poroelastic_edge);

//   h_elastic_ispec = elastic_poroelastic.elastic_ispec;
//   h_poroelastic_ispec = elastic_poroelastic.poroelastic_ispec;

//   ASSERT(h_elastic_ispec.extent(0) == num_interfaces, "Invalid
//   elastic_ispec"); ASSERT(h_poroelastic_ispec.extent(0) == num_interfaces,
//          "Invalid poroelastic_ispec");
//   ASSERT(elastic_edge.extent(0) == num_interfaces, "Invalid elastic_edge");
//   ASSERT(h_poroelastic_edge.extent(0) == num_interfaces,
//          "Invalid poroelastic_edge");

// #ifndef NDEBUG
//   for (int i = 0; i < num_interfaces; i++) {
//     int ispec_elastic = h_elastic_ispec(i);
//     int ispec_poroelastic = h_poroelastic_ispec(i);

//     // Check that the interface is between an elastic and a poroelastic
//     // element
//     ASSERT(((ispec_elastic != ispec_poroelastic) &&
//             (h_ispec_type(ispec_elastic) ==
//              specfem::element::medium_tag::elastic_sv) &&
//             (h_ispec_type(ispec_poroelastic) ==
//              specfem::element::medium_tag::poroelastic)),
//            "Invalid interface");
//   }
// #endif

//   compute_edges(h_ibool, h_elastic_ispec, h_poroelastic_ispec,
//   h_elastic_edge,
//                 h_poroelastic_edge);

// #ifndef NDEBUG
//   check_edges(h_ibool, coord, h_elastic_ispec, h_poroelastic_ispec,
//               h_elastic_edge, h_poroelastic_edge);
// #endif

//   Kokkos::deep_copy(elastic_ispec, h_elastic_ispec);
//   Kokkos::deep_copy(poroelastic_ispec, h_poroelastic_ispec);
//   Kokkos::deep_copy(elastic_edge, h_elastic_edge);
//   Kokkos::deep_copy(poroelastic_edge, h_poroelastic_edge);

//   return;
// }

// specfem::compute::coupled_interfaces::acoustic_poroelastic::
//     acoustic_poroelastic(
//         const specfem::kokkos::HostMirror3d<int> h_ibool,
//         const specfem::kokkos::HostView2d<type_real> coord,
//         const specfem::kokkos::HostView1d<specfem::enums::element::type>
//             h_ispec_type,
//         const specfem::mesh::coupled_interfaces::acoustic_poroelastic
//             &acoustic_poroelastic)
//     : num_interfaces(acoustic_poroelastic.num_interfaces),
//       acoustic_ispec(
//           "compute::coupled_interfaces::acoustic_poroelastic::acoustic_ispec",
//           acoustic_poroelastic.num_interfaces),
//       poroelastic_ispec("compute::coupled_interfaces::acoustic_poroelastic::"
//                         "poroelastic_ispec",
//                         acoustic_poroelastic.num_interfaces),
//       acoustic_edge(
//           "compute::coupled_interfaces::acoustic_poroelastic::acoustic_edge",
//           acoustic_poroelastic.num_interfaces),
//       poroelastic_edge("compute::coupled_interfaces::acoustic_poroelastic::"
//                        "poroelastic_edge",
//                        acoustic_poroelastic.num_interfaces) {

//   if (num_interfaces == 0)
//     return;

//   h_acoustic_ispec = Kokkos::create_mirror_view(acoustic_ispec);
//   h_poroelastic_ispec = Kokkos::create_mirror_view(poroelastic_ispec);
//   h_acoustic_edge = Kokkos::create_mirror_view(acoustic_edge);
//   h_poroelastic_edge = Kokkos::create_mirror_view(poroelastic_edge);

//   h_acoustic_ispec = acoustic_poroelastic.acoustic_ispec;
//   h_poroelastic_ispec = acoustic_poroelastic.poroelastic_ispec;

//   ASSERT(h_acoustic_ispec.extent(0) == num_interfaces,
//          "Invalid acoustic_ispec");
//   ASSERT(h_poroelastic_ispec.extent(0) == num_interfaces,
//          "Invalid poroelastic_ispec");
//   ASSERT(acoustic_edge.extent(0) == num_interfaces, "Invalid acoustic_edge");
//   ASSERT(h_poroelastic_edge.extent(0) == num_interfaces,
//          "Invalid poroelastic_edge");

// #ifndef NDEBUG
//   for (int i = 0; i < num_interfaces; i++) {
//     int ispec_acoustic = h_acoustic_ispec(i);
//     int ispec_poroelastic = h_poroelastic_ispec(i);

//     // Check that the interface is between an acoustic and a poroelastic
//     // element
//     ASSERT(((ispec_acoustic != ispec_poroelastic) &&
//             (h_ispec_type(ispec_acoustic) ==
//              specfem::element::medium_tag::acoustic) &&
//             (h_ispec_type(ispec_poroelastic) ==
//              specfem::element::medium_tag::poroelastic)),
//            "Invalid interface");
//   }
// #endif

//   compute_edges(h_ibool, h_acoustic_ispec, h_poroelastic_ispec,
//   h_acoustic_edge,
//                 h_poroelastic_edge);

// #ifndef NDEBUG
//   check_edges(h_ibool, coord, h_acoustic_ispec, h_poroelastic_ispec,
//               h_acoustic_edge, h_poroelastic_edge);
// #endif

//   Kokkos::deep_copy(acoustic_ispec, h_acoustic_ispec);
//   Kokkos::deep_copy(poroelastic_ispec, h_poroelastic_ispec);
//   Kokkos::deep_copy(acoustic_edge, h_acoustic_edge);
//   Kokkos::deep_copy(poroelastic_edge, h_poroelastic_edge);

//   return;
// }

// specfem::compute::coupled_interfaces::coupled_interfaces::coupled_interfaces(
//     const specfem::kokkos::HostMirror3d<int> h_ibool,
//     const specfem::kokkos::HostView2d<type_real> coord,
//     const specfem::kokkos::HostView1d<specfem::enums::element::type>
//         h_ispec_type,
//     const specfem::mesh::coupled_interfaces::coupled_interfaces
//         &coupled_interfaces)
//     :
//     elastic_acoustic(specfem::compute::coupled_interfaces::elastic_acoustic(
//           h_ibool, coord, h_ispec_type,
//           coupled_interfaces.elastic_acoustic)),
//       elastic_poroelastic(
//           specfem::compute::coupled_interfaces::elastic_poroelastic(
//               h_ibool, coord, h_ispec_type,
//               coupled_interfaces.elastic_poroelastic)),
//       acoustic_poroelastic(
//           specfem::compute::coupled_interfaces::acoustic_poroelastic(
//               h_ibool, coord, h_ispec_type,
//               coupled_interfaces.acoustic_poroelastic)) {}

specfem::compute::coupled_interfaces::coupled_interfaces(
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::compute::points &points,
    const specfem::compute::quadrature &quadrature,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::element_types &element_types,
    const specfem::compute::mesh_to_compute_mapping &mapping)
    : elastic_acoustic(mesh, points, quadrature, partial_derivatives,
                       element_types, mapping),
      elastic_poroelastic(mesh, points, quadrature, partial_derivatives,
                          element_types, mapping),
      acoustic_poroelastic(mesh, points, quadrature, partial_derivatives,
                           element_types, mapping) {}

// Explicit template instantiation

template class specfem::compute::interface_container<
    specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::acoustic>;

template class specfem::compute::interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>;

template class specfem::compute::interface_container<
    specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::poroelastic>;

template specfem::compute::interface_container<
    specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::acoustic>
specfem::compute::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::acoustic>() const;

template specfem::compute::interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_sv>
specfem::compute::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_sv>() const;

template specfem::compute::interface_container<
    specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::poroelastic>
specfem::compute::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::poroelastic>() const;

template specfem::compute::interface_container<
    specfem::element::medium_tag::poroelastic,
    specfem::element::medium_tag::elastic_sv>
specfem::compute::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::poroelastic,
    specfem::element::medium_tag::elastic_sv>() const;

template specfem::compute::interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>
specfem::compute::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>() const;

template specfem::compute::interface_container<
    specfem::element::medium_tag::poroelastic,
    specfem::element::medium_tag::acoustic>
specfem::compute::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::poroelastic,
    specfem::element::medium_tag::acoustic>() const;

// Explicit template member function instantiation
