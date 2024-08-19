#ifndef DOMAIN_IMPL_ELEMENTS_KERNEL_HPP
#define DOMAIN_IMPL_ELEMENTS_KERNEL_HPP

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
class element_kernel_base {

public:
  inline int total_elements() const { return nelements; }

  using dimension = specfem::dimension::dimension<DimensionType>;
  // using element_type = specfem::domain::impl::elements::element<
  //     WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
  //     quadrature_points_type>;
  using medium_type = specfem::medium::medium<DimensionType, MediumTag>;
  using qp_type = quadrature_points_type;

  element_kernel_base() = default;
  element_kernel_base(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping,
      const quadrature_points_type &quadrature_points);

  void compute_mass_matrix(
      const type_real dt,
      const specfem::compute::simulation_field<WavefieldType> &field) const;

  void compute_stiffness_interaction(
      const int istep,
      const specfem::compute::simulation_field<WavefieldType> &field) const;

  template <specfem::enums::time_scheme::type time_scheme>
  void mass_time_contribution(
      const type_real dt,
      const specfem::compute::simulation_field<WavefieldType> &field) const {};

protected:
  int nelements;
  specfem::compute::points points;
  specfem::compute::quadrature quadrature;
  specfem::kokkos::DeviceView1d<int> element_kernel_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_element_kernel_index_mapping;
  specfem::compute::properties properties;
  specfem::compute::partial_derivatives partial_derivatives;
  specfem::compute::boundaries boundaries;
  specfem::compute::boundary_value_container<DimensionType, BoundaryTag>
      boundary_values;
  quadrature_points_type quadrature_points;
  // element_type element;
};

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
class element_kernel
    : public element_kernel_base<WavefieldType, DimensionType, MediumTag,
                                 PropertyTag, BoundaryTag,
                                 quadrature_points_type> {

public:
  element_kernel() = default;
  element_kernel(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping,
      const quadrature_points_type &quadrature_points)
      : field(assembly.fields.get_simulation_field<WavefieldType>()),
        element_kernel_base<WavefieldType, DimensionType, MediumTag,
                            PropertyTag, BoundaryTag, quadrature_points_type>(
            assembly, h_element_kernel_index_mapping, quadrature_points) {}

  void compute_mass_matrix(const type_real dt) const {
    element_kernel_base<WavefieldType, DimensionType, MediumTag, PropertyTag,
                        BoundaryTag,
                        quadrature_points_type>::compute_mass_matrix(dt, field);
  }

  void compute_stiffness_interaction(const int istep) const {
    element_kernel_base<
        WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
        quadrature_points_type>::compute_stiffness_interaction(istep, field);
  }

  template <specfem::enums::time_scheme::type time_scheme>
  void mass_time_contribution(const type_real dt) const {
    element_kernel_base<WavefieldType, DimensionType, MediumTag, PropertyTag,
                        BoundaryTag, quadrature_points_type>::
        template mass_time_contribution<time_scheme>(dt, field);
  }

private:
  specfem::compute::simulation_field<WavefieldType> field;
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          typename quadrature_points_type>
class element_kernel<
    specfem::wavefield::type::backward, DimensionType, MediumTag, PropertyTag,
    specfem::element::boundary_tag::stacey, quadrature_points_type>
    : public element_kernel_base<specfem::wavefield::type::backward,
                                 DimensionType, MediumTag, PropertyTag,
                                 specfem::element::boundary_tag::stacey,
                                 quadrature_points_type> {

public:
  element_kernel() = default;
  element_kernel(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping,
      const quadrature_points_type &quadrature_points)
      : field(assembly.fields
                  .get_simulation_field<specfem::wavefield::type::backward>()),
        element_kernel_base<specfem::wavefield::type::backward, DimensionType,
                            MediumTag, PropertyTag,
                            specfem::element::boundary_tag::stacey,
                            quadrature_points_type>(
            assembly, h_element_kernel_index_mapping, quadrature_points) {}

  void compute_mass_matrix(const type_real dt) const {
    element_kernel_base<specfem::wavefield::type::backward, DimensionType,
                        MediumTag, PropertyTag,
                        specfem::element::boundary_tag::stacey,
                        quadrature_points_type>::compute_mass_matrix(dt, field);
  }

  template <specfem::enums::time_scheme::type time_scheme>
  void mass_time_contribution(const type_real dt) const {
    element_kernel_base<specfem::wavefield::type::backward, DimensionType,
                        MediumTag, PropertyTag,
                        specfem::element::boundary_tag::stacey,
                        quadrature_points_type>::
        template mass_time_contribution<time_scheme>(dt, field);
  }

  void compute_stiffness_interaction(const int istep) const;

private:
  specfem::compute::simulation_field<specfem::wavefield::type::backward> field;
};

} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // DOMAIN_IMPL_ELEMENTS_KERNEL_HPP
