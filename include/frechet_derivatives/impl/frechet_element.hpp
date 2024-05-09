#ifndef _FRECHET_DERIVATIVES_IMPL_FRECHLET_ELEMENT_HPP
#define _FRECHET_DERIVATIVES_IMPL_FRECHLET_ELEMENT_HPP

#include "compute/assembly/assembly.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/fields/impl/field_impl.hpp"
#include "compute/kernels/kernels.hpp"
#include "compute/properties/properties.hpp"

namespace specfem {
namespace frechet_derivatives {
namespace impl {
template <int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
class frechet_elements {
public:
  frechet_elements(const specfem::compute::assembly &assembly);

  void compute(const type_real &dt);

private:
  specfem::kokkos::DeviceView1d<int> element_index;
  specfem::kokkos::HostMirror1d<int> h_element_index;
  specfem::compute::simulation_field<specfem::wavefield::type::adjoint>
      adjoint_field;
  specfem::compute::simulation_field<specfem::wavefield::type::backward>
      backward_field;
  specfem::compute::kernels kernels;
  specfem::compute::quadrature quadrature;
  specfem::compute::properties properties;
  specfem::compute::partial_derivatives partial_derivatives;
};
} // namespace impl
} // namespace frechet_derivatives
} // namespace specfem

#endif /* _FRECHET_DERIVATIVES_IMPL_FRECHLET_ELEMENT_HPP */
