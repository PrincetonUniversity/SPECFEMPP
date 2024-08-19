#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/coupled_interfaces/interface_container.hpp"
#include "compute/properties/properties.hpp"
#include "edge/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace compute {

namespace impl {

struct interface_index {
  int medium1_index;
  int medium2_index;
};

} // namespace impl

template <specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2>
struct interface_container {

private:
  using IndexView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using EdgeTypeView =
      Kokkos::View<specfem::enums::edge::type *, Kokkos::DefaultExecutionSpace>;
  using EdgeFactorView = Kokkos::View<type_real **, Kokkos::LayoutRight,
                                      Kokkos::DefaultExecutionSpace>;
  using EdgeNormalView = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                      Kokkos::DefaultExecutionSpace>;

public:
  constexpr static specfem::element::medium_tag medium1_type = MediumTag1;
  constexpr static specfem::element::medium_tag medium2_type = MediumTag2;

  interface_container() = default;

  interface_container(const int num_interfaces, const int ngll);

  interface_container(
      const specfem::mesh::mesh &mesh, const specfem::compute::points &points,
      const specfem::compute::quadrature &quadrature,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties,
      const specfem::compute::mesh_to_compute_mapping &mapping);

  interface_container(const interface_container<MediumTag2, MediumTag1> &other)
      : num_interfaces(other.num_interfaces), num_points(other.num_points),
        medium1_index_mapping(other.medium2_index_mapping),
        h_medium1_index_mapping(other.h_medium2_index_mapping),
        medium2_index_mapping(other.medium1_index_mapping),
        h_medium2_index_mapping(other.h_medium1_index_mapping),
        medium1_edge_type(other.medium2_edge_type),
        h_medium1_edge_type(other.h_medium2_edge_type),
        medium2_edge_type(other.medium1_edge_type),
        h_medium2_edge_type(other.h_medium1_edge_type),
        medium1_edge_factor(other.medium2_edge_factor),
        h_medium1_edge_factor(other.h_medium2_edge_factor),
        medium2_edge_factor(other.medium1_edge_factor),
        h_medium2_edge_factor(other.h_medium1_edge_factor),
        medium1_edge_normal(other.medium2_edge_normal),
        h_medium1_edge_normal(other.h_medium2_edge_normal),
        medium2_edge_normal(other.medium1_edge_normal),
        h_medium2_edge_normal(other.h_medium1_edge_normal) {
    return;
  }

  int num_interfaces;
  int num_points;
  IndexView medium1_index_mapping;
  IndexView medium2_index_mapping;

  EdgeTypeView medium1_edge_type;
  EdgeTypeView medium2_edge_type;

  IndexView::HostMirror h_medium1_index_mapping;
  IndexView::HostMirror h_medium2_index_mapping;

  EdgeTypeView::HostMirror h_medium1_edge_type;
  EdgeTypeView::HostMirror h_medium2_edge_type;

  EdgeFactorView medium1_edge_factor;
  EdgeFactorView medium2_edge_factor;

  EdgeFactorView::HostMirror h_medium1_edge_factor;
  EdgeFactorView::HostMirror h_medium2_edge_factor;

  EdgeNormalView medium1_edge_normal;
  EdgeNormalView medium2_edge_normal;

  EdgeNormalView::HostMirror h_medium1_edge_normal;
  EdgeNormalView::HostMirror h_medium2_edge_normal;

  std::tuple<IndexView, IndexView> get_index_mapping() const {
    return std::make_tuple(medium1_index_mapping, medium2_index_mapping);
  }

  std::tuple<EdgeTypeView, EdgeTypeView> get_edge_type() const {
    return std::make_tuple(medium1_edge_type, medium2_edge_type);
  }

  EdgeFactorView get_edge_factor() const { return medium1_edge_factor; }

  EdgeNormalView get_edge_normal() const { return medium1_edge_normal; }
};
} // namespace compute
} // namespace specfem

// namespace specfem {
// namespace compute {
// template <specfem::element::medium_tag medium1,
//           specfem::element::medium_tag medium2>
// struct interface_container {

//   constexpr static specfem::element::medium_tag medium1_type = medium1;
//   constexpr static specfem::element::medium_tag medium2_type = medium2;

//   interface_container() = default;

//   interface_container(const specfem::compute::mesh &mesh,
//                       const specfem::compute::properties &properties,
//                       const specfem::mesh::interface_container<medium1,
//                       medium2>
//                           &coupled_interfaces);

//   interface_container(const interface_container<medium2, medium1> &other)
//       : num_interfaces(other.num_interfaces),
//         medium1_index_mapping(other.medium2_index_mapping),
//         h_medium1_index_mapping(other.h_medium2_index_mapping),
//         medium2_index_mapping(other.medium1_index_mapping),
//         h_medium2_index_mapping(other.h_medium1_index_mapping),
//         medium1_edge_type(other.medium2_edge_type),
//         h_medium1_edge_type(other.h_medium2_edge_type),
//         medium2_edge_type(other.medium1_edge_type),
//         h_medium2_edge_type(other.h_medium1_edge_type) {
//     return;
//   }

//   int num_interfaces = 0;

//   specfem::kokkos::DeviceView1d<int> medium1_index_mapping; ///< spectral
//                                                             ///< element
//                                                             number
//                                                             ///< for the ith
//                                                             ///< edge in
//                                                             medium
//                                                             ///< 1

//   specfem::kokkos::HostMirror1d<int> h_medium1_index_mapping; ///< spectral
//                                                               ///< element
//                                                               ///< number for
//                                                               ///< the ith
//                                                               edge
//                                                               ///< in medium
//                                                               1

//   specfem::kokkos::DeviceView1d<int> medium2_index_mapping; ///< spectral
//                                                             ///< element
//                                                             number
//                                                             ///< for the ith
//                                                             ///< element in
//                                                             ///< medium 2

//   specfem::kokkos::HostMirror1d<int> h_medium2_index_mapping; ///< spectral
//                                                               ///< element
//                                                               ///< number for
//                                                               ///< the ith
//                                                               ///< element in
//                                                               ///< medium 2

//   specfem::kokkos::DeviceView1d<specfem::edge::interface>
//       medium1_edge_type; ///< edge type for the ith edge in medium 1
//   specfem::kokkos::HostMirror1d<specfem::edge::interface>
//       h_medium1_edge_type; ///< edge type for the ith edge in medium 1

//   specfem::kokkos::DeviceView1d<specfem::edge::interface>
//       medium2_edge_type; ///< edge type for the ith edge in medium 2
//   specfem::kokkos::HostMirror1d<specfem::edge::interface>
//       h_medium2_edge_type; ///< edge type for the ith edge in medium 2

//   template <specfem::element::medium_tag medium>
//   KOKKOS_FUNCTION int load_device_index_mapping(const int iedge) const;

//   template <specfem::element::medium_tag medium>
//   int load_host_index_mapping(const int iedge) const;

//   template <specfem::element::medium_tag medium>
//   KOKKOS_FUNCTION specfem::edge::interface load_device_edge_type(
//       const int iedge) const;

//   template <specfem::element::medium_tag medium>
//   specfem::edge::interface load_host_edge_type(const int iedge) const;
// };
// } // namespace compute
// } // namespace specfem
