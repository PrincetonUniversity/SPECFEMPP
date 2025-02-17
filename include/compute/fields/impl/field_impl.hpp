#ifndef _COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_
#define _COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_

#include "compute/compute_mesh.hpp"
#include "compute/element_types/element_types.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
class field_impl {
public:
  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components();

  field_impl() = default;

  field_impl(
      const specfem::compute::mesh &mesh,
      const specfem::compute::element_types &element_type,
      Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
          assembly_index_mapping);

  field_impl(const int nglob);

  template <specfem::sync::kind sync> void sync_fields() const;

  int nglob;
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

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
void deep_copy(impl::field_impl<DimensionType, MediumTag> &dst,
               const impl::field_impl<DimensionType, MediumTag> &src) {
  Kokkos::deep_copy(dst.field, src.field);
  Kokkos::deep_copy(dst.h_field, src.h_field);
  Kokkos::deep_copy(dst.field_dot, src.field_dot);
  Kokkos::deep_copy(dst.h_field_dot, src.h_field_dot);
  Kokkos::deep_copy(dst.field_dot_dot, src.field_dot_dot);
  Kokkos::deep_copy(dst.h_field_dot_dot, src.h_field_dot_dot);
}

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_ */
