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

  static_assert(std::is_same_v<medium, typename BC::medium_type>,
                "Boundary conditions should have the same medium type as the "
                "element kernel");
  static_assert(std::is_same_v<dimension, typename BC::dimension>,
                "Boundary conditions should have the same dimension as the "
                "element kernel");
  static_assert(std::is_same_v<qp_type, typename BC::quadrature_points_type>,
                "Boundary conditions should have the same quadrature point "
                "type as the element kernel");
  static_assert(std::is_same_v<property, typename BC::property_type>,
                "Boundary conditions should have the same property as the "
                "element kernel");

  element_kernel() = default;
  element_kernel(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::DeviceView1d<int> element_kernel_index_mapping,
      const quadrature_point_type &quadrature_points);

  void compute_mass_matrix() const;

  void compute_stiffness_interaction() const;

  template <specfem::enums::time_scheme::type time_scheme>
  void mass_time_contribution(const type_real dt) const;

  inline int total_elements() const { return nelements; }

private:
  int nelements;
  specfem::compute::points points;
  specfem::compute::quadrature quadrature;
  specfem::kokkos::DeviceView1d<int> element_kernel_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_element_kernel_index_mapping;
  Kokkos::View<int * [specfem::enums::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::DevMemSpace>
      global_index_mapping;
  specfem::compute::properties properties;
  specfem::compute::partial_derivatives partial_derivatives;
  specfem::kokkos::DeviceView1d<specfem::point::boundary> boundary_conditions;
  specfem::compute::impl::field_impl<medium_type> field;
  quadrature_point_type quadrature_points;
  specfem::domain::impl::elements::element<
      dimension, medium, qp_type, property_type, boundary_conditions_type>
      element;

  // using load_properties = properties.load_properties<medium_type::value,
  // property_type::value, specfem::kokkos::DevExecSpace>;

  // template <bool load_jacobian>
  // using load_partial_derivatives =
  // partial_derivatives.load_partial_derivatives<load_jacobian,
  // specfem::kokkos::DevExecSpace>;
};

} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // DOMAIN_IMPL_ELEMENTS_KERNEL_HPP
