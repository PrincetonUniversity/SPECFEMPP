#ifndef _DOMAIN_IMPL_SOURCES_KERNEL_HPP
#define _DOMAIN_IMPL_SOURCES_KERNEL_HPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/interface.hpp"
#include "domain/impl/sources/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <class medium, class qp_type, typename... elemental_properties>
class source_kernel {
public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium_type = medium;
  using quadrature_point_type = qp_type;

  source_kernel() = default;
  source_kernel(const specfem::kokkos::DeviceView3d<int> ibool,
                const specfem::kokkos::DeviceView1d<int> ispec,
                const specfem::kokkos::DeviceView1d<int> isource,
                const specfem::compute::properties &properties,
                const specfem::compute::sources &sources,
                quadrature_point_type quadrature_points,
                specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
                    field_dot_dot);

  void compute_source_interaction(const type_real timeval) const;

private:
  specfem::kokkos::DeviceView1d<int> ispec;
  specfem::kokkos::DeviceView3d<int> ibool;
  specfem::kokkos::DeviceView1d<int> isource;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot;
  specfem::kokkos::DeviceView1d<specfem::forcing_function::stf_storage>
      stf_array;
  quadrature_point_type quadrature_points;
  specfem::domain::impl::sources::source<
      dimension, medium_type, quadrature_point_type, elemental_properties...>
      source;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // _DOMAIN_IMPL_SOURCES_KERNEL_HPP
