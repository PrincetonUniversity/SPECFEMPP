#ifndef _COMPUTE_COUPLED_INTERFACES_INTERFACE_CONTAINER_HPP
#define _COMPUTE_COUPLED_INTERFACES_INTERFACE_CONTAINER_HPP

#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "edge/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace compute {
template <specfem::enums::element::type medium1,
          specfem::enums::element::type medium2>
struct interface_container {

  constexpr static specfem::enums::element::type medium1_type = medium1;
  constexpr static specfem::enums::element::type medium2_type = medium2;

  interface_container() = default;

  interface_container(const specfem::compute::mesh &mesh,
                      const specfem::compute::properties &properties,
                      const specfem::mesh::interface_container<medium1, medium2>
                          &coupled_interfaces);

  int num_interfaces = 0;

  specfem::kokkos::DeviceView1d<int> medium1_index_mapping; ///< spectral
                                                            ///< element number
                                                            ///< for the ith
                                                            ///< edge in medium
                                                            ///< 1

  specfem::kokkos::HostMirror1d<int> h_medium1_index_mapping; ///< spectral
                                                              ///< element
                                                              ///< number for
                                                              ///< the ith edge
                                                              ///< in medium 1

  specfem::kokkos::DeviceView1d<int> medium2_index_mapping; ///< spectral
                                                            ///< element number
                                                            ///< for the ith
                                                            ///< element in
                                                            ///< medium 2

  specfem::kokkos::HostMirror1d<int> h_medium2_index_mapping; ///< spectral
                                                              ///< element
                                                              ///< number for
                                                              ///< the ith
                                                              ///< element in
                                                              ///< medium 2

  specfem::kokkos::DeviceView1d<specfem::edge::interface>
      medium1_edge_type; ///< edge type for the ith edge in medium 1
  specfem::kokkos::HostMirror1d<specfem::edge::interface>
      h_medium1_edge_type; ///< edge type for the ith edge in medium 1

  specfem::kokkos::DeviceView1d<specfem::edge::interface>
      medium2_edge_type; ///< edge type for the ith edge in medium 2
  specfem::kokkos::HostMirror1d<specfem::edge::interface>
      h_medium2_edge_type; ///< edge type for the ith edge in medium 2

  template <specfem::enums::element::type medium>
  specfem::kokkos::DeviceView1d<int> get_index_mapping_view() const;

  template <specfem::enums::element::type medium>
  specfem::kokkos::DeviceView1d<specfem::edge::interface>
  get_edge_type_view() const;
};
} // namespace compute
} // namespace specfem

#endif
