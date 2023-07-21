#ifndef _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_HPP_
#define _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_HPP_

#include "domain/impl/receivers/receiver.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {

template <class quadrature_points>
class receiver<specfem::enums::element::dimension::dim2,
               specfem::enums::element::medium::elastic, quadrature_points> {
public:
  using value_type = type_real[];
  KOKKOS_INLINE_FUNCTION virtual void get_field(
      const int xz, const specfem::kokkos::DeviceView2d<type_real> field,
      const specfem::kokkos::DeviceView2d<type_real> field_dot,
      const specfem::kokkos::DeviceView2d<type_real> field_dot_dot) const {};
  KOKKOS_INLINE_FUNCTION virtual void compute_seismogram_components(
      const int xz, type_real (&l_seismogram_components)[2]) const {};
  KOKKOS_INLINE_FUNCTION virtual void
  compute_seismogram(const int isig_step,
                     const type_real (&seismogram_components)[2]) const {};
  KOKKOS_INLINE_FUNCTION virtual specfem::enums::seismogram::type
  get_seismogram_type() const = 0;

  using medium = specfem::enums::element::medium::elastic;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif /* _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_HPP_ */
