#ifndef _DOMAIN_KERNELS_HPP
#define _DOMAIN_KERNELS_HPP

#include "compute/interface.hpp"
#include "domain/impl/elements/interface.hpp"
#include "domain/impl/sources/interface.hpp"
#include "specfem_enums.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <class medium, class qp_type> class kernels {

public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium_type = medium;
  using quadrature_point_type = qp_type;

  kernels() = default;

  kernels(
      const specfem::kokkos::DeviceView3d<int> ibool,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties,
      const specfem::compute::sources &sources, quadrature::quadrature *quadx,
      quadrature::quadrature *quadz, qp_type quadrature_points,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot_dot,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> mass_matrix);

  __inline__ void compute_stiffness_interaction() const {
    isotropic_elements.compute_stiffness_interaction();
    return;
  }

  __inline__ void compute_mass_matrix() const {
    isotropic_elements.compute_mass_matrix();
    return;
  }

  __inline__ void compute_source_interaction(const type_real timeval) const {
    isotropic_sources.compute_source_interaction(timeval);
    return;
  }

private:
  qp_type quadrature_points;
  specfem::domain::impl::kernels::element_kernel<
      medium_type, quadrature_point_type,
      specfem::enums::element::property::isotropic>
      isotropic_elements;
  specfem::domain::impl::kernels::source_kernel<
      medium_type, quadrature_point_type,
      specfem::enums::element::property::isotropic>
      isotropic_sources;
  //   specfem::domain::kernels::receivers<
  //       dimension, medium_type, quadrature_points_type,
  //       specfem::enums::element::property::isotropic>
  //       isotropic_receivers;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // _DOMAIN_KERNELS_HPP
