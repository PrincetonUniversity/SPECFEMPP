#ifndef _COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_
#define _COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_

#include "compute/compute_mesh.hpp"
#include "compute/properties/interface.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {
template <typename medium> class field_impl {
public:
  using medium_type = medium;

  constexpr static int components = medium::components;

  field_impl() = default;

  field_impl(const specfem::compute::mesh &mesh,
             const specfem::compute::properties &properties,
             Kokkos::View<int * [specfem::enums::element::ntypes],
                          Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
                 assembly_index_mapping);

  field_impl(const int nglob, const int nspec, const int ngllz,
             const int ngllx);

  int nglob;
  int nspec;
  int ngllz;
  int ngllx;
  specfem::kokkos::DeviceView3d<type_real> index_mapping;
  specfem::kokkos::HostMirror3d<type_real> h_index_mapping;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field;
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft> h_field;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot;
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft> h_field_dot;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot;
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft> h_field_dot_dot;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> mass_inverse;
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft> h_mass_inverse;
};
} // namespace impl

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_ */
