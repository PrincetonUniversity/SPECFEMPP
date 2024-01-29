#ifndef _COMPUTE_BOUNDARIES_IMPL_BOUNDARY_CONTAINER_HPP
#define _COMPUTE_BOUNDARIES_IMPL_BOUNDARY_CONTAINER_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "point/boundary.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {

namespace boundaries {
template <specfem::enums::element::boundary_tag boundary_tag>
struct boundary_container {

  constexpr static specfem::enums::element::boundary_tag value =
      boundary_tag; ///< Boundary tag

  boundary_container() = default;

  boundary_container(
      const std::vector<specfem::enums::element::boundary_tag_container>
          &boundary_tags,
      const std::vector<specfem::point::boundary> &boundary_types);

  int nelements;
  specfem::kokkos::DeviceView1d<int> index_mapping;
  specfem::kokkos::HostMirror1d<int> h_index_mapping;
  specfem::kokkos::DeviceView1d<specfem::point::boundary> boundary_type;
  specfem::kokkos::HostMirror1d<specfem::point::boundary> h_boundary_type;
};
} // namespace boundaries
} // namespace impl
} // namespace compute
} // namespace specfem

#endif
