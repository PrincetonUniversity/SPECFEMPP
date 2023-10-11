#ifndef _DOMAIN_IMPL_RECEIVERS_KERNEL_HPP
#define _DOMAIN_IMPL_RECEIVERS_KERNEL_HPP

#include "domain/impl/receivers/acoustic/interface.hpp"
#include "domain/impl/receivers/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <class medium, class qp_type, typename... elemental_properties>
class receiver_kernel {
public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium_type = medium;
  using quadrature_points_type = qp_type;

  receiver_kernel() = default;

  receiver_kernel(
      const specfem::kokkos::DeviceView3d<int> ibool,
      const specfem::kokkos::DeviceView1d<int> ispec,
      const specfem::kokkos::DeviceView1d<int> ireceiver,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties,
      const specfem::compute::receivers &receivers,
      const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
      const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot,
      const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot_dot,
      specfem::quadrature::quadrature *quadx,
      specfem::quadrature::quadrature *quadz,
      quadrature_points_type quadrature_points);

  void compute_seismograms(const int &isig_step) const;

private:
  specfem::kokkos::DeviceView3d<int> ibool;
  specfem::kokkos::DeviceView1d<int> ispec;
  specfem::kokkos::DeviceView1d<int> ireceiver;
  specfem::kokkos::DeviceView1d<int> iseis;
  specfem::kokkos::DeviceView1d<specfem::enums::seismogram::type>
      seismogram_types;
  specfem::kokkos::DeviceView4d<type_real> receiver_seismogram;
  quadrature_points_type quadrature_points;
  specfem::quadrature::quadrature *quadx;
  specfem::quadrature::quadrature *quadz;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot;
  specfem::domain::impl::receivers::receiver<
      dimension, medium_type, quadrature_points_type, elemental_properties...>
      receiver;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // _DOMAIN_IMPL_RECEIVERS_KERNEL_HPP
