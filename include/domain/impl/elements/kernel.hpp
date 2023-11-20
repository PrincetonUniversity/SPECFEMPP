#ifndef DOMAIN_IMPL_ELEMENTS_KERNEL_HPP
#define DOMAIN_IMPL_ELEMENTS_KERNEL_HPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/interface.hpp"
#include "domain/impl/elements/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <class medium, class qp_type, typename property, typename BC>
class element_kernel {

public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium_type = medium;
  using quadrature_point_type = qp_type;
  using property_type = property;
  using boundary_conditions_type = BC;

  element_kernel() = default;
  element_kernel(
      const specfem::kokkos::DeviceView3d<int> ibool,
      const specfem::kokkos::DeviceView1d<int> ispec,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties,
      const specfem::compute::boundaries &boundary_conditions,
      specfem::quadrature::quadrature *quadx,
      specfem::quadrature::quadrature *quadz, qp_type quadrature_points,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot_dot,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> mass_matrix);

  void compute_mass_matrix() const;

  void compute_stiffness_interaction() const;

  template <specfem::enums::time_scheme::type time_scheme>
  void mass_time_contribution(const type_real dt) const;

private:
  specfem::kokkos::DeviceView1d<int> ispec;
  specfem::kokkos::DeviceView3d<int> ibool;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> mass_matrix;
  qp_type quadrature_points;
  specfem::quadrature::quadrature *quadx;
  specfem::quadrature::quadrature *quadz;
  specfem::domain::impl::elements::element<
      dimension, medium, qp_type, property_type, boundary_conditions_type>
      element;
};

} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // DOMAIN_IMPL_ELEMENTS_KERNEL_HPP
