#ifndef _DOMIAN_IMPL_RECEIVERS_ACOUSTIC2D_HPP_
#define _DOMIAN_IMPL_RECEIVERS_ACOUSTIC2D_HPP_

#include "constants.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {

template <class qp_type>
class receiver<specfem::enums::element::dimension::dim2,
               specfem::enums::element::medium::acoustic, qp_type> {
public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium = specfem::enums::element::medium::acoustic;
  using quadrature_points = qp_type;
  using value_type = type_real[];
  KOKKOS_INLINE_FUNCTION virtual void get_field(
      const int xz, const int isig_step,
      const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
      const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot,
      const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot_dot) const {};
  KOKKOS_INLINE_FUNCTION virtual void compute_seismogram_components(
      const int xz, const int isig_step,
      dimension::array_type<type_real> &l_seismogram_components) const {};
  KOKKOS_INLINE_FUNCTION virtual void compute_seismogram(
      const int isig_step,
      const dimension::array_type<type_real> &seismogram_components){};
  KOKKOS_INLINE_FUNCTION virtual specfem::enums::seismogram::type
  get_seismogram_type() const = 0;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif /* _DOMIAN_IMPL_RECEIVERS_ACOUSTIC2D_HPP_ */
